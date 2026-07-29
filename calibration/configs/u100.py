# calibration 모듈 config — U100 차량 제원 (PIPELINE.md §8 매트릭스 그대로).

config = dict(
    vehicle="U100",
    width_m=1.89,
    wheelbase_m=2.68,
    front_length_m=3.45,
    rear_length_m=1.265,
    steer_gain=[8.3629, 15, 100],  # PIPELINE.md §8 표기 "8.3629/15/100" 그대로
    imu_lat_offset_m=0.1593,
)
