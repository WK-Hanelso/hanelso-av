from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from planning.interface import Policy, register_policy


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(value) for value in obj)
    return obj


class PlutoTorchPolicy(Policy):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but torch.cuda.is_available() is False"
            )
        self.model, self.model_kwargs, self.load_report = self._load_model()

    def _load_model(self):
        from planning.models.pluto.src.models.pluto.pluto_model import PlanningModel

        with self.config_path.open() as f:
            config = yaml.safe_load(f)
        model_cfg = dict(config["model"])
        model_cfg.pop("feature_builder", None)
        kwargs = {
            key: value
            for key, value in model_cfg.items()
            if key not in {"_target_", "_convert_"}
        }

        torch.manual_seed(0)
        model = PlanningModel(**kwargs)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key[6:] if key.startswith("model.") else key: value
            for key, value in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        report = {
            "checkpoint_key_count": len(state_dict),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "strict_match": not missing and not unexpected,
        }
        if missing or unexpected:
            raise RuntimeError(
                "Strict PlanningModel load failed: "
                f"missing={missing} unexpected={unexpected}"
            )
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        model.to(self.device)
        report["decoder_swap"] = {
            "enabled": False,
            "decoder": "planning_decoder_merged_v3",
            "load_state_dict_missing_keys": [],
            "load_state_dict_unexpected_keys": [],
            "strict_match": True,
        }
        report["device"] = str(self.device)
        report["model_param_device"] = str(next(model.parameters()).device)
        return model, kwargs, report

    @torch.inference_mode()
    def infer(self, feature: Any) -> dict[str, Any]:
        if hasattr(feature, "to_device"):
            feature = feature.to_device(self.device)
        batch = feature.data if hasattr(feature, "data") else feature
        output = self.model(batch)
        output = _to_cpu(output)
        return {
            "output_trajectory": output["output_trajectory"],
            "candidate_trajectories": output["candidate_trajectories"],
            "probability": output["probability"],
            "output_prediction": output["output_prediction"],
            "raw_output": output,
        }


register_policy("pluto_torch", PlutoTorchPolicy)
