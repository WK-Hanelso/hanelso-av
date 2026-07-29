# planning/models/ — 모델-우선 배치 규약

모델의 **모든 코드는 `models/<이름>/` 한 디렉토리가 소유**한다 (docker "이미지=모델 소유"와
동일 원칙). 드라이버(`run_inference.py`, `simulation/render_sim.py`, `build_input.py`)는
concrete를 모른다 — root config `modules.planning` 이름으로
`planning.interface.load_model(이름)`이 `planning.models.<이름>` 패키지를 import하고,
그 `__init__`이 registry에 자기 구현을 등록한다. 없는 이름이면
`Unknown planning model 'xxx'. Available: [...]` (이 디렉토리 스캔).

## 모델 추가 규약

**models/<이름>/ 디렉토리 1개 + `planning/configs/<이름>.py` 1개 — 기존 파일 수정 0.**

- `__init__.py`에서 하위 모듈 import → `planning.interface`의 register 호출.
- dataloader는 `REQUIRES`(data_devkit 아티팩트 이름)를 선언하고
  `data_devkit.contract.check()`로 fail-fast 검증.
- 모듈 config가 registry 키(policy/input_builder/feed_builder/postprocess.name)를 정한다.

## 현재 모델

| 디렉토리 | 파일 | registry 키 |
|---|---|---|
| `pluto/` | `policy.py`(PlutoTorchPolicy, strict ckpt 로드) · `dataloader.py`(ApolloPlutoFeatureAdapter + REQUIRES/check) · `input_builder.py`(plain numpy feed) · `postprocess.py`(원본 TrajectoryEvaluator+EmergencyBrake) · `v3_planning_decoder.py` · `paths.py`(vendored third_party/pluto sys.path 일원화) | policy=`pluto_torch`, adapter=`pluto_feature`, feed=`pluto`, post=`pluto` |
