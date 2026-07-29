# calibration

차량 제원(치수·STEER_GAIN·IMU offset) 도메인. `configs/{e100,u100}.py`는
PIPELINE.md §8 매트릭스의 사실 기록이고, 소비 시점 배선은
`calibration/vehicle.py`가 담당한다.

현재 배선 상태:
- `to_vehicle_parameters(calib)`가 nuPlan `VehicleParameters`를 만든다.
- `steering_pct_to_tire_angle()` / `max_tire_angle()`가 `% -> tire angle rad`
  변환을 맡는다.
- `ego_dims()` / `rear_axle_to_center()`가 물리 계층(충돌 check, sim render,
  forward simulation) 치수를 제공한다.
- IMU 횡오프셋 보정은 파서가 아니라 dataloader 소비 시점에서 적용한다.

소유권 원칙:
- calibration은 실차 제원만 소유한다.
- feature ego-shape pin은 학습 분포의 속성이므로 root config가 아니라
  `planning/configs/pluto.py`의 `feature_vehicle`가 소유한다.
