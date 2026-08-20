# data_devkit/counterfactual — 반사실 goal 생성기 (FORMAT_SPEC §11 생산자)

"실제와 다른 goal + 그리로 가는 것이 정답인 궤적" 쌍을 생성한다. goal 조건 채널이
자연 데이터에서 redundant해져 죽는 것(설계 원장 §7 — stock 모델 명령 무시 실측으로 입증)을
막는 학습 데이터. **스키마·게이트의 SoT는 `common/FORMAT_SPEC.md` §11** (운영 문서, 로컬 전용) —
이 모듈은 그 구현이며 필드 추가·의미 변경을 하지 않는다.

## 구성

| 파일 | 역할 |
|---|---|
| `generator.py` | 개입 2종(`lane_transplant` ±3.5m quintic / `time_transplant`) + §11.1 품질 게이트(곡률≤0.2, 횡가속≤4.0 — 위반은 valid=false로 보존) + §11.2 vocab 커버리지(`VocabCoverage`, 선택) + §11 스키마 emit. **torch-free** (data_devkit 규약), 토큰은 md5 결정론(재실행 안정) |
| `generate_openscene.py` | OpenScene 로그(pkl) 소스 CLI 드라이버 — `counterfactual.json` + `report.json`(생성 리포트: 유효율·커버리지 분포) 산출 |

## 실행 (항상 docker run)

```bash
docker run --rm -v "$PWD":/workspace -w /workspace \
  -v /mnt/hdd_storage/openscene_mini:/data:ro av-base:latest \
  python -m data_devkit.counterfactual.generate_openscene \
    --logs /data/mini_navsim_logs/mini \
    --out work/counterfactual/openscene_mini \
    --vocab data/model/sparsedrive_v2/kmeans/trajectory_1024_256.npz
```

## 검증 실적 (OpenScene mini, 2026-08-20)

64로그 → 앵커 2,848 → 반사실 7,608개, valid 88.5%, vocab 최근접 거리 p50 0.30m / p90 0.47m
— 반사실 정답이 scoring vocabulary 커버리지 안에 있음(§11.2 성립).

## 확장 예정 (설계 확정분만)

- 맵 스냅 lane_transplant: 횡 offset을 실제 인접 차선 중심선으로 (§11 `intervention.params` 확장)
- `command_flip` / `sim_collected` 타입: 시뮬 개입 데이터 엔진(회의 결정 8-②, Bench2Drive)과 합류
