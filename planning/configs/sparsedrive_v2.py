# planning 모듈 config — SparseDriveV2 (System 1 온보딩 슬롯, 1단계).
# 원칙은 pluto.py와 동일: 학습이 결정한 값은 bundle 내 native config에서 읽는다(전사 금지).
# registry 키·bundle 파일명은 온보딩 3~5단계(반입·계약 편입)에서 확정 — 아래는 자리 예약.
# 명세 SoT: planning/models/sparsedrive_v2/README.md

config = dict(
    status="onboarding_stage1",        # registry 미구현 — 조립 시 fail-fast (__init__.py)
    policy=None,                       # 예약: policy registry key (5단계 확정)
    bundle="data/model/sparsedrive_v2",  # 아티팩트 묶음 규약 (ckpt + native config 쌍)
    model_config=None,                 # 예약: bundle 내 native config 파일명
    checkpoint=None,                   # 예약: bundle 내 checkpoint 파일명
    device="cuda",                     # System 1은 GPU 전제 (목표 20Hz)
    env="sparsedrivev2-inf",           # 이 모델의 docker 이미지 (docker/sparsedrivev2-inf/)
    postprocess=dict(enabled=False, name=None),  # V2는 scoring이 본체 — 별도 postprocessor 없음(#30 optional 계약)
    dataloader=None,                   # 예약: raw tier(카메라) 계약과 함께 5단계 확정
)
