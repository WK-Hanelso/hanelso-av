from __future__ import annotations

import math
from typing import Dict, Tuple

# nuPlan OrientedBox paths expect a concrete height. We do not have a measured
# E100/U100 roof height in §8, so keep the existing Pacifica-compatible height.
DEFAULT_VEHICLE_HEIGHT_M = 1.777


def _required_float(calib: Dict[str, object], key: str) -> float:
    if key not in calib:
        raise KeyError(f"Missing calibration key: {key}")
    return float(calib[key])


def to_vehicle_parameters(calib: Dict[str, object]):
    """Build a nuPlan VehicleParameters from calibration §8 values."""
    from nuplan.common.actor_state.vehicle_parameters import VehicleParameters

    width = _required_float(calib, "width_m")
    front = _required_float(calib, "front_length_m")
    rear = _required_float(calib, "rear_length_m")
    wheel_base = _required_float(calib, "wheelbase_m")
    vehicle_name = str(calib["vehicle"])

    # Keep the CoG derivation simple and calibration-local: absent a measured CoG,
    # place it at the wheelbase midpoint instead of inheriting Pacifica ratios.
    cog_position = wheel_base / 2.0

    return VehicleParameters(
        width=width,
        front_length=front,
        rear_length=rear,
        cog_position_from_rear_axle=cog_position,
        wheel_base=wheel_base,
        vehicle_name=vehicle_name,
        vehicle_type="custom",
        height=DEFAULT_VEHICLE_HEIGHT_M,
    )


def max_tire_angle(calib: Dict[str, object]) -> float:
    steer_gain = calib.get("steer_gain")
    if not isinstance(steer_gain, (list, tuple)) or len(steer_gain) != 3:
        raise ValueError("calibration.steer_gain must be [max_wheel_rad, steering_ratio, pct_scale]")
    max_wheel_rad, steering_ratio, pct_scale = (float(value) for value in steer_gain)
    if steering_ratio == 0.0 or pct_scale == 0.0:
        raise ValueError("steer_gain steering_ratio/pct_scale must be non-zero")
    return abs(max_wheel_rad / steering_ratio)


def steering_pct_to_tire_angle(pct: float, calib: Dict[str, object]) -> float:
    steer_gain = calib.get("steer_gain")
    if not isinstance(steer_gain, (list, tuple)) or len(steer_gain) != 3:
        raise ValueError("calibration.steer_gain must be [max_wheel_rad, steering_ratio, pct_scale]")
    max_wheel_rad, steering_ratio, pct_scale = (float(value) for value in steer_gain)
    if steering_ratio == 0.0 or pct_scale == 0.0:
        raise ValueError("steer_gain steering_ratio/pct_scale must be non-zero")
    tire_angle = float(pct) * max_wheel_rad / steering_ratio / pct_scale
    max_angle = max_tire_angle(calib)
    return float(max(-max_angle, min(max_angle, tire_angle)))


def ego_dims(calib: Dict[str, object]) -> Tuple[float, float]:
    width = _required_float(calib, "width_m")
    front = _required_float(calib, "front_length_m")
    rear = _required_float(calib, "rear_length_m")
    return width, front + rear


def rear_axle_to_center(calib: Dict[str, object]) -> float:
    params = to_vehicle_parameters(calib)
    return float(params.rear_axle_to_center)


def apply_imu_lateral_offset(
    positions_xy,
    headings,
    imu_lat_offset_m: float,
    sign: float,
):
    import numpy as np

    if imu_lat_offset_m == 0.0:
        return np.asarray(positions_xy, dtype=np.float64).copy()
    headings_np = np.asarray(headings, dtype=np.float64)
    positions_np = np.asarray(positions_xy, dtype=np.float64).copy()
    left_unit = np.stack([-np.sin(headings_np), np.cos(headings_np)], axis=-1)
    positions_np += float(sign) * float(imu_lat_offset_m) * left_unit
    return positions_np
