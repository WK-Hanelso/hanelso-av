from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch
import yaml

from planning.models.pluto.paths import ensure_pluto_on_path
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
        pluto_root: Optional[Union[str, Path]] = None,
        use_v3_planning_decoder: bool = True,
        device: str = "cpu",
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.pluto_root = ensure_pluto_on_path(pluto_root)
        self.use_v3_planning_decoder = use_v3_planning_decoder
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but torch.cuda.is_available() is False"
            )
        self.model, self.model_kwargs, self.load_report = self._load_model()

    def _load_model(self):
        from src.models.pluto.pluto_model import PlanningModel
        from .v3_planning_decoder import PlanningDecoder as V3PlanningDecoder

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

        decoder_name = "original"
        decoder_swap_report = {
            "enabled": bool(self.use_v3_planning_decoder),
            "decoder": decoder_name,
        }
        if self.use_v3_planning_decoder:
            v3_decoder = V3PlanningDecoder(
                num_mode=kwargs["num_modes"],
                decoder_depth=kwargs["decoder_depth"],
                dim=kwargs["dim"],
                num_heads=kwargs["num_heads"],
                mlp_ratio=4,
                dropout=kwargs["dropout"],
                future_steps=kwargs["future_steps"],
                cat_x=kwargs.get("cat_x", False),
            )
            swap_missing, swap_unexpected = v3_decoder.load_state_dict(
                model.planning_decoder.state_dict(), strict=True
            )
            model.planning_decoder = v3_decoder
            decoder_name = "v3_fixed"
            decoder_swap_report = {
                "enabled": True,
                "decoder": decoder_name,
                "load_state_dict_missing_keys": list(swap_missing),
                "load_state_dict_unexpected_keys": list(swap_unexpected),
                "strict_match": not swap_missing and not swap_unexpected,
            }

        model.eval()
        model.to(self.device)
        report["decoder_swap"] = decoder_swap_report
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
