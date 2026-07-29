# planning/models/ — 모델-우선 배치 규약

모델의 **모든 코드는 `models/<이름>/` 한 디렉토리가 소유**한다 (docker "이미지=모델 소유"와
동일 원칙). 드라이버(`run_inference.py`, `simulation/render_sim.py`)는
concrete를 모른다 — root config `modules.planning` 이름으로
`planning.interface.load_model(이름)`이 `planning.models.<이름>` 패키지를 import하고,
그 `__init__`이 registry에 자기 구현을 등록한다. 없는 이름이면
`Unknown planning model 'xxx'. Available: [...]` (이 디렉토리 스캔).

## 모델 추가 규약

**models/<이름>/ 디렉토리 1개 + `planning/configs/<이름>.py` 1개 — 기존 파일 수정 0.**

- `__init__.py`에서 하위 모듈 import → `planning.interface`의 register 호출.
- dataloader는 `REQUIRES`(data_devkit 아티팩트 이름)를 선언하고
  `data_devkit.contract.check()`로 fail-fast 검증.
- 모듈 config가 registry 키(policy/dataloader/postprocess.name)를 정한다.

## 모델 온보딩 절차

1. 반입: upstream 모델 소스를 워킹트리로 가져온다.
2. 분해: 모델 전용 코드는 `models/<이름>/src/`, 공용부는 해당 도메인으로 승격한다.
3. adapter: `policy.py`, `dataloader.py`, 필요 시 `postprocess.py`를 붙인다.
4. config: `planning/configs/<이름>.py`에 registry 키와 bundle/env만 기록한다.
5. bundle: `data/model/<이름>_vN/`에 checkpoint + native config 쌍을 둔다.
6. docker: `docker/<이름>-inf/`에 실추론 이미지를 둔다.

모델 소스는 모델 홈(`models/<이름>/src/`)에서 직접 수정한다. 버전 실험이 필요하면
overlay 파일을 만들지 말고 `src/` 안에 클래스를 병존시키고 모델 config 선택으로 갈라진다.

## 현재 모델

| 디렉토리 | 파일 | registry 키 |
|---|---|---|
| `pluto/` | `policy.py`(PlutoTorchPolicy, strict ckpt 로드) · `dataloader.py`(ApolloPlutoDataloader + REQUIRES/check + feed-building body 포함) · `postprocess.py`(원본 TrajectoryEvaluator+EmergencyBrake 어댑터) · `src/`(PLUTO 모델 원본+학습 하네스 직접 소유) | policy=`pluto_torch`, dataloader=`pluto_feature`, post=`pluto` |
