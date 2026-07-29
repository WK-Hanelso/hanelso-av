# calibration 모듈 config — E100 차량 제원 (PIPELINE.md §8 매트릭스 그대로).
# 이번 태스크는 데이터(제원 config)까지만; 코드 배선(rear-axle 변환·STEER_GAIN·
# 충돌 shape 적용)은 다음 태스크.

config = dict(
    vehicle="E100",
    width_m=1.87,
    wheelbase_m=2.67,
    front_length_m=3.325,          # front len (PIPELINE.md §8 "front/rear len")
    rear_length_m=1.135,           # rear len
    steer_gain=[9.42478, 15, 100],  # PIPELINE.md §8 표기 "9.42478/15/100" 그대로
    imu_lat_offset_m=0.1652,
)
