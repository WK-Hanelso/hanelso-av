# simulation 모듈 config — closed-loop(모델 운전) × 원본 nuplan 렌더러.
# mode -> ego-driver registry: open_loop=log_replay / closed_loop=model_driven.
# sim은 오프라인 검증 경로라 CPU(swm-base)로 돈다. GPU 실추론 env는 모델 소유(pluto-inf).

config = dict(
    mode="closed_loop",   # open_loop | closed_loop
    renderer="nuplan",    # matplotlib | nuplan (simulation/renderers registry)
    steps=100,            # sim steps (closed) / rendered frames (open)
    stride=2,             # open_loop: rendered frame당 진행할 log frame 수
    fps=None,             # mp4 fps (None -> open: 10/stride, closed: 10)
    start_index=None,     # t_start log index (None -> 첫 full-history frame)
    view_radius=50.0,
    postprocess=True,
    env="swm-base",       # 공용 파이프라인(파싱·sim·render) docker 이미지
)
