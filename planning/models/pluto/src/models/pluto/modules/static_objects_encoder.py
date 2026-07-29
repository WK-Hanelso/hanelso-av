import math
import torch
import torch.nn as nn

from ..layers.fourier_embedding import FourierEmbedding


class StaticObjectsEncoder(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.obj_encoder = FourierEmbedding(2, dim, 64)
        self.type_emb = nn.Embedding(4, dim)

        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.01)

    def forward(self, data):
        pos = data["static_objects"]["position"]
        heading = data["static_objects"]["heading"]
        shape = data["static_objects"]["shape"]
        category = data["static_objects"]["category"].long()
        valid_mask = data["static_objects"]["valid_mask"]  # [bs, N]

        # ONNX-safe static-shape refactor (P1/B2d).
        # Intent: encode every static object densely, then zero-out invalid
        # (padded) objects so they contribute nothing downstream. obj_encoder is
        # a FourierEmbedding (Linear/LayerNorm/ReLU, NO BatchNorm) and type_emb
        # is an Embedding, both per-row independent, so the dense compute is
        # already what the original does; the ONLY masking is zeroing invalid
        # rows. Replace the size-changing boolean index-assign with a
        # mask-multiply (identical result). Handles N==0-valid gracefully: an
        # all-invalid batch simply yields an all-zero obj_emb, and the returned
        # ``~valid_mask`` key-padding mask makes the downstream attention ignore
        # every zeroed row anyway.
        obj_emb_tmp = self.obj_encoder(shape) + self.type_emb(category.long())
        obj_emb = obj_emb_tmp * valid_mask.unsqueeze(-1).to(obj_emb_tmp.dtype)

        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        obj_pos = torch.cat([pos, heading.unsqueeze(-1)], dim=-1)

        return obj_emb, obj_pos, ~valid_mask
