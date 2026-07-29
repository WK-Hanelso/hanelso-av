# simulation 모듈 config — open-loop(로그 ego 재생) × matplotlib BEV 렌더러.
# 공통 필드는 closed_loop_nuplan.py에서 상속(_base_)하고 차이만 덮어쓴다.

_base_ = "closed_loop_nuplan.py"

config = dict(
    mode="open_loop",
    renderer="matplotlib",
)
