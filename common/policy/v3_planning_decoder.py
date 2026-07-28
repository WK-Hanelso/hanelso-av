from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.pluto.layers.embedding import PointsEncoder
from src.models.pluto.layers.fourier_embedding import FourierEmbedding
from src.models.pluto.layers.mlp_layer import MLPLayer


# Original reference:
# /home/hanelso/hanelso/pluto_onnx/src/models/pluto/modules/planning_decoder.py
# Sole delta vs original: r2r_attn key_padding_mask repeat -> repeat_interleave.


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout) -> None:
        super().__init__()
        self.dim = dim

        self.r2r_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.m2m_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        m_pos: Optional[Tensor] = None,
    ):
        """
        tgt: (bs, R, M, dim)
        tgt_key_padding_mask: (bs, R)
        """
        bs, R, M, D = tgt.shape

        tgt = tgt.transpose(1, 2).reshape(bs * M, R, D)
        tgt2 = self.norm1(tgt)
        # ONNX-export note (P1/B4): under export force the decomposed MHA path
        # (distinct k/v objects, need_weights=False). Inert at eval/runtime
        # (guarded on is_in_onnx_export()), so numerics are unchanged.
        if torch.onnx.is_in_onnx_export():
            r2r_k = r2r_v = tgt2 + 0.0
            tgt2 = self.r2r_attn(
                tgt2,
                r2r_k,
                r2r_v,
                key_padding_mask=tgt_key_padding_mask.repeat_interleave(M, dim=0),
                need_weights=False,
            )[0]
        else:
            tgt2 = self.r2r_attn(
                tgt2,
                tgt2,
                tgt2,
                key_padding_mask=tgt_key_padding_mask.repeat_interleave(M, dim=0),
            )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt_tmp = tgt.reshape(bs, M, R, D).transpose(1, 2).reshape(bs * R, M, D)
        tgt_valid_mask = ~tgt_key_padding_mask.reshape(-1)  # (bs*R,)

        # ONNX-safe static-shape refactor of the m2m (mode-to-mode) attention
        # (P1/B3, the hard one).
        #
        # Original: gather ONLY the valid reference-line rows out of tgt_tmp
        # (shape (bs*R, M, D)), run the mode-attention over that variable-sized
        # subset, then scatter results back into an all-zero (bs*R, M, D) tensor.
        #
        # Key insight: m2m_attn is a *batched* self-attention whose batch axis is
        # (bs*R) and whose sequence axis is the mode axis M. It has NO
        # key_padding_mask and mixes information only WITHIN a row (across modes),
        # never across rows. So each row is processed fully independently ->
        # running the attention densely over ALL bs*R rows produces, for every
        # VALID row, exactly the same output as gathering first (LayerNorm is
        # also per-row). Invalid rows can't corrupt valid ones, and because there
        # is no key_padding_mask no row is ever fully masked -> no NaN. We then
        # zero the invalid rows via a mask-multiply, reproducing the original
        # zeros_like+scatter bit-for-bit for the valid rows.
        #
        # Training keeps the ORIGINAL gather path verbatim: dropout2 draws its
        # random mask over a tensor whose size depends on the number of valid
        # rows, so running it densely in training would change the RNG
        # consumption and thus training numerics. In eval/export dropout is an
        # identity, so the dense path is numerically identical (verified
        # allclose).
        if self.training and not torch.onnx.is_in_onnx_export():
            tgt_valid = tgt_tmp[tgt_valid_mask]
            tgt2_valid = self.norm2(tgt_valid)
            tgt2_valid, _ = self.m2m_attn(
                tgt2_valid + m_pos, tgt2_valid + m_pos, tgt2_valid
            )
            tgt_valid = tgt_valid + self.dropout2(tgt2_valid)
            tgt = torch.zeros_like(tgt_tmp)
            tgt[tgt_valid_mask] = tgt_valid
        else:
            tgt2 = self.norm2(tgt_tmp)
            tgt2, _ = self.m2m_attn(tgt2 + m_pos, tgt2 + m_pos, tgt2)
            tgt_dense = tgt_tmp + self.dropout2(tgt2)
            # zero out invalid reference-line rows (matches zeros_like+scatter)
            valid = tgt_valid_mask.view(-1, 1, 1).to(tgt_dense.dtype)  # (bs*R,1,1)
            tgt = tgt_dense * valid

        tgt = tgt.reshape(bs, R, M, D).view(bs, R * M, D)
        tgt2 = self.norm3(tgt)
        tgt2 = self.cross_attn(
            tgt2, memory, memory, key_padding_mask=memory_key_padding_mask
        )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm4(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = tgt.reshape(bs, R, M, D)

        return tgt


class PlanningDecoder(nn.Module):
    def __init__(
        self,
        num_mode,
        decoder_depth,
        dim,
        num_heads,
        mlp_ratio,
        dropout,
        future_steps,
        yaw_constraint=False,
        cat_x=False,
    ) -> None:
        super().__init__()

        self.num_mode = num_mode
        self.future_steps = future_steps
        self.yaw_constraint = yaw_constraint
        self.cat_x = cat_x

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderLayer(dim, num_heads, mlp_ratio, dropout)
                for _ in range(decoder_depth)
            ]
        )

        self.r_pos_emb = FourierEmbedding(3, dim, 64)
        self.r_encoder = PointsEncoder(6, dim)

        self.q_proj = nn.Linear(2 * dim, dim)

        self.m_emb = nn.Parameter(torch.Tensor(1, 1, num_mode, dim))
        self.m_pos = nn.Parameter(torch.Tensor(1, num_mode, dim))

        if self.cat_x:
            self.cat_x_proj = nn.Linear(2 * dim, dim)

        self.loc_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.yaw_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.vel_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.pi_head = MLPLayer(dim, dim, 1)

        nn.init.normal_(self.m_emb, mean=0.0, std=0.01)
        nn.init.normal_(self.m_pos, mean=0.0, std=0.01)

    def forward(self, data, enc_data):
        enc_emb = enc_data["enc_emb"]
        enc_key_padding_mask = enc_data["enc_key_padding_mask"]

        r_position = data["reference_line"]["position"]
        r_vector = data["reference_line"]["vector"]
        r_orientation = data["reference_line"]["orientation"]
        r_valid_mask = data["reference_line"]["valid_mask"]
        r_key_padding_mask = ~r_valid_mask.any(-1)

        r_feature = torch.cat(
            [
                r_position - r_position[..., 0:1, :2],
                r_vector,
                torch.stack([r_orientation.cos(), r_orientation.sin()], dim=-1),
            ],
            dim=-1,
        )

        bs, R, P, C = r_feature.shape
        r_valid_mask = r_valid_mask.view(bs * R, P)
        r_feature = r_feature.reshape(bs * R, P, C)
        r_emb = self.r_encoder(r_feature, r_valid_mask).view(bs, R, -1)

        r_pos = torch.cat([r_position[:, :, 0], r_orientation[:, :, 0, None]], dim=-1)
        r_emb = r_emb + self.r_pos_emb(r_pos)

        r_emb = r_emb.unsqueeze(2).repeat(1, 1, self.num_mode, 1)
        m_emb = self.m_emb.repeat(bs, R, 1, 1)

        q = self.q_proj(torch.cat([r_emb, m_emb], dim=-1))

        for blk in self.decoder_blocks:
            q = blk(
                q,
                enc_emb,
                tgt_key_padding_mask=r_key_padding_mask,
                memory_key_padding_mask=enc_key_padding_mask,
                m_pos=self.m_pos,
            )
            # Value-assert forces a host<->device sync and is not exportable.
            # Skip it under ONNX export; keep it for normal eager runs.
            if not torch.onnx.is_in_onnx_export():
                assert torch.isfinite(q).all()

        if self.cat_x:
            x = enc_emb[:, 0].unsqueeze(1).unsqueeze(2).repeat(1, R, self.num_mode, 1)
            q = self.cat_x_proj(torch.cat([q, x], dim=-1))

        loc = self.loc_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        yaw = self.yaw_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        vel = self.vel_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        pi = self.pi_head(q).squeeze(-1)

        traj = torch.cat([loc, yaw, vel], dim=-1)

        return traj, pi
