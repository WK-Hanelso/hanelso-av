"""PLUTO PlanningModel variant matching ``last.ckpt`` (plantf-style pos_emb).

The trained checkpoint (``pluto_onnx/last.ckpt``, 25 epochs) is the PLUTO
architecture EXCEPT for the positional embedding, which was trained PlanTF-style:

    ckpt: pos_emb.0 = Linear(4 -> dim), pos_emb.2 = Linear(dim -> dim)
          => nn.Sequential(Linear(4, dim), ReLU(), Linear(dim, dim))
          == plantf build_mlp(4, [dim, dim])

whereas the stock ``PlanningModel`` uses ``FourierEmbedding(3, dim, 64)`` (a 3-D
[x, y, angle] input). All other 403 parameters are identical (verified by a
strict state_dict load).

Consequently every token position fed to ``pos_emb`` must be the 4-D PlanTF
representation ``[x, y, cos(theta), sin(theta)]`` (agents, polygons AND static
objects). This subclass leaves the stock model untouched and only:
  1. swaps ``self.pos_emb`` for the 4-D MLP, and
  2. re-implements ``forward`` with 4-D pos construction.
Everything else (encoders, decoder, eval post-processing) is copied verbatim.
"""
import math

import torch
import torch.nn as nn

from .pluto_model import PlanningModel


class PlanningModelPlantfPos(PlanningModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Replace the FourierEmbedding(3, dim, 64) with the plantf-style 4-D MLP
        # (build_mlp(4, [dim, dim], activation="relu")): Linear(4,dim), ReLU,
        # Linear(dim,dim). Indices .0 and .2 match the checkpoint keys exactly.
        self.pos_emb = nn.Sequential(
            nn.Linear(4, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )
        # Re-init only the freshly created pos_emb (xavier + zero bias), matching
        # the model's global _init_weights convention.
        self.pos_emb.apply(self._init_weights)

    @staticmethod
    def _angle_to_cos_sin(angle: torch.Tensor) -> torch.Tensor:
        """(..., ) angle -> (..., 2) [cos, sin]. plantf convention."""
        return torch.stack([angle.cos(), angle.sin()], dim=-1)

    def forward(self, data):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        # 4-D plantf pos: [x, y, cos(theta), sin(theta)]. cos/sin are periodic so
        # the (angle+pi)%2pi-pi normalization used by the FourierEmbedding path is
        # unnecessary here and intentionally omitted (matches plantf training).
        pos = torch.cat([position, self._angle_to_cos_sin(angle)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        x = torch.cat([x_agent, x_polygon, x_static], dim=1)

        # static_objects_encoder returns obj_pos = [x, y, heading] (3-D). Convert
        # to the same 4-D [x, y, cos, sin] representation before concatenation.
        static_pos = torch.cat(
            [static_pos[..., :2], self._angle_to_cos_sin(static_pos[..., 2])], dim=-1
        )

        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)

        prediction = self.agent_predictor(x[:, 1:A])

        # Export assumption: reference lines are always present (R > 0).
        ref_line_available = True

        if ref_line_available:
            trajectory, probability = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}
            )
        else:
            trajectory, probability = None, None

        out = {
            "trajectory": trajectory,
            "probability": probability,  # (bs, R, M)
            "prediction": prediction,  # (bs, A-1, T, 2)
        }

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])

        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )
            out["ref_free_trajectory"] = ref_free_traj

        if not self.training:
            if self.ref_free_traj:
                ref_free_traj_angle = torch.arctan2(
                    ref_free_traj[..., 3], ref_free_traj[..., 2]
                )
                ref_free_traj = torch.cat(
                    [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
                )
                out["output_ref_free_trajectory"] = ref_free_traj

            output_prediction = torch.cat(
                [
                    prediction[..., :2] + agent_pos[:, 1:A, None],
                    torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                    + agent_heading[:, 1:A, None, None],
                    prediction[..., 4:6],
                ],
                dim=-1,
            )
            out["output_prediction"] = output_prediction

            if trajectory is not None:
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ]

                out["output_trajectory"] = best_trajectory
                out["candidate_trajectories"] = out_trajectory
            else:
                out["output_trajectory"] = out["output_ref_free_trajectory"]
                out["probability"] = torch.zeros(1, 0, 0)
                out["candidate_trajectories"] = torch.zeros(
                    1, 0, 0, self.future_steps, 3
                )

        return out
