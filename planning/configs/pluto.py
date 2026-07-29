# planning 모듈 config — PLUTO. 3요소(policy/bundle/native config 참조) + env.
# 원칙: 학습이 결정한 값(모델 하이퍼파라미터)은 bundle 내 native config
# (hydra 산출 config.yaml)에서 읽는다 — 여기에 전사 금지.

config = dict(
    policy="pluto_torch",              # planning.interface policy registry key
    bundle="data/model/pluto_v3",      # 아티팩트 묶음(불투명 디렉토리, ckpt+native config 쌍)
    model_config="config.yaml",        # bundle 내 native config — 경로 참조만
    checkpoint="v3_pluto.ckpt",        # bundle 내 checkpoint
    device="cpu",                      # "cpu"|"cuda" — root/CLI에서 override 가능
    env="pluto-inf",                   # 이 모델의 docker 이미지(실추론용; sim/CPU는 swm-base)
    postprocess=dict(enabled=True, name="pluto"),  # postprocessor registry key
    input_builder="pluto_feature",     # feature adapter(dataloader) registry key
)

# build_input.py(plain numpy feed 드라이버)가 쓰는 InputBuilder registry key.
config["feed_builder"] = "pluto"
