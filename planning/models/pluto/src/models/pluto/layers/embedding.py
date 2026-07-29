import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# P1/B1: NATTEN's NeighborhoodAttention1D uses a custom CUDA op with no ONNX
# symbolic. Swap to a pure-PyTorch, ONNX-exportable, numerically-equivalent
# reimplementation. Param names/shapes (qkv, proj, rpb) match NATTEN 1:1, so any
# NATTEN-trained checkpoint loads unchanged. No `natten` import remains here.
from .native_nat import NativeNeighborhoodAttention1D as NeighborhoodAttention1D


class NATSequenceEncoder(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dim=32,
        mlp_ratio=3,
        kernel_size=[3, 3, 5],
        depths=[2, 2, 2],
        num_heads=[2, 4, 8],
        out_indices=[0, 1, 2],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.embed = ConvTokenizer(in_chans, embed_dim)
        self.num_levels = len(depths)
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_levels)]
        self.out_indices = out_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size[i],
                dilations=None,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
            )
            self.levels.append(level)

        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        n = self.num_features[-1]
        self.lateral_convs = nn.ModuleList()
        for i_layer in self.out_indices:
            self.lateral_convs.append(
                nn.Conv1d(self.num_features[i_layer], n, 3, padding=1)
            )

        self.fpn_conv = nn.Conv1d(n, n, 3, padding=1)

    def forward(self, x):
        """x: [B, C, T]"""
        x = self.embed(x)

        out = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                out.append(x_out.permute(0, 2, 1).contiguous())

        laterals = [
            lateral_conv(out[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(out) - 1, 0, -1):
            # ONNX-export note (P1): use explicit output `size=` instead of a
            # Python-float `scale_factor=`. torch 1.12's ONNX tracer rejects a
            # float scale_factor passed to upsample_linear1d. `size` is the
            # target length (== the source level's length, i.e. exactly what the
            # scale_factor ratio computed), so this is bit-identical to the
            # original while being trace/export-friendly.
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-1],
                mode="linear",
                align_corners=False,
            )

        out = self.fpn_conv(laterals[0])

        return out[:, :, -1]


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=32, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 1)  # B, C, L -> B, L, C
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv1d(
            dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention1D(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class PointsEncoder(nn.Module):
    def __init__(self, feat_channel, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_mlp = nn.Sequential(
            nn.Linear(feat_channel, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.second_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.encoder_channel),
        )

    def forward(self, x, mask=None):
        """
        x : B M 3
        mask: B M
        -----------------
        feature_global : B C
        """

        bs, n, _ = x.shape
        device = x.device

        # ONNX-safe static-shape refactor (P1/B2b).
        # Intent: per-polygon, encode ONLY valid points; aggregate them via
        # max-pool; padded points must contribute ZERO. The two MLPs contain
        # BatchNorm1d: in TRAINING the original runs BN over the *valid points
        # only*, so its running-stats update from that subset. Running BN densely
        # over all (incl. padded) rows in training would change which samples
        # feed the running stats -> different training behavior. Therefore we
        # keep the ORIGINAL boolean-index path VERBATIM for training and use a
        # static-shape equivalent only for eval / ONNX export. In eval, BN uses
        # frozen running stats, so dense-then-zero is numerically identical to
        # the gathered path (verified allclose).
        #
        # Note on the max-pool: the ORIGINAL pools over a ZERO-FILLED tensor
        # (invalid rows are 0 via the scatter), so the padding value 0 competes
        # in the max. We replicate that EXACTLY (zero-fill invalid rows, then
        # max) rather than masked_fill(-inf), because -inf would change outputs
        # whenever all valid features at a channel are negative. Preserving the
        # original numerics is the overriding requirement.
        if self.training and not torch.onnx.is_in_onnx_export():
            # ----- original path (verbatim), kept bit-identical for training -----
            x_valid = self.first_mlp(x[mask])  # B n 256
            x_features = torch.zeros(bs, n, 256, device=device)
            x_features[mask] = x_valid

            pooled_feature = x_features.max(dim=1)[0]
            x_features = torch.cat(
                [x_features, pooled_feature.unsqueeze(1).repeat(1, n, 1)], dim=-1
            )

            x_features_valid = self.second_mlp(x_features[mask])
            res = torch.zeros(bs, n, self.encoder_channel, device=device)
            res[mask] = x_features_valid

            res = res.max(dim=1)[0]
            return res

        # ----- static-shape eval / export path -----
        m = mask.unsqueeze(-1).to(x.dtype)  # B n 1

        x_features = self.first_mlp(x.reshape(bs * n, -1)).view(bs, n, 256)
        x_features = x_features * m  # zero out padded points (matches scatter)

        pooled_feature = x_features.max(dim=1)[0]  # pool over zero-filled tensor
        x_features = torch.cat(
            [x_features, pooled_feature.unsqueeze(1).repeat(1, n, 1)], dim=-1
        )

        res = self.second_mlp(x_features.reshape(bs * n, -1)).view(
            bs, n, self.encoder_channel
        )
        res = res * m  # zero out padded points before final max (matches scatter)

        res = res.max(dim=1)[0]
        return res
