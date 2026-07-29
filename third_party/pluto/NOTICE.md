# Vendored: PLUTO (src/)

이 디렉토리(`src/`)는 **PLUTO** 소스를 hanelso_swm 내부로 vendor한 것이다.
프로젝트가 외부 경로(`--pluto-root`)에 의존하지 않고 `git clone` 하나로 재현되도록 내부에 포함한다.

- 원본: PLUTO — *Push the Limit of Imitation Learning-based Planning for Autonomous Driving* (공식 repo, jchengai/pluto 계열).
- 반입 버전: `pluto_onnx` 트리(= NATTEN을 순수 torch로 재구현한 `native_nat.py` 포함, CPU/ONNX 친화). NATTEN 미설치로 CPU forward 가능하며, 원본 NATTEN과 forward 출력이 `max_abs_diff ~1e-6`로 수치 동등함을 검증함.
- hanelso_swm 코드는 `sys.path`에 이 디렉토리를 추가해 `from src...`로 import한다(기본 `--pluto-root = third_party/pluto`).

## 라이선스
원본 로컬 트리에 LICENSE 파일이 없었다(README만 존재). PLUTO 공식 배포는 통상 Apache-2.0이다.
**배포/공개 전 원저작자 라이선스(권장: 원 repo의 LICENSE)를 이 디렉토리에 포함하고 준수 여부를 확정할 것.** (원저작권·수정사항 고지 유지.)

## 로컬 수정
- 없음(반입 그대로). 우리 측 디코더 수정본은 원본을 건드리지 않고 `common/policy/v3_planning_decoder.py`에 별도 존재.
