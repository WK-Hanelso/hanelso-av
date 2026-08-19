# planning/models/sparsedrive_v2 — System 1 온보딩 명세 (1단계: 자리잡기)

이중 시스템 아키텍처(설계 판단 원장)의 **System 1**(실시간 궤적 계층, 목표 20Hz) 기성품 온보딩 슬롯.
상태: **온보딩 중 — registry 미구현** (import 시 명확한 에러로 안내). 이 README가 후속 단계의 기준 명세다.

## 기성품

| 항목 | 값 |
|---|---|
| upstream | [swc-17/SparseDriveV2](https://github.com/swc-17/SparseDriveV2) — 코드·가중치 공개 |
| 라이선스 | **Apache-2.0** (반입 시 `NOTICE.md` 고지 필수 — third_party 규약) |
| 성능 근거 | Bench2Drive 폐루프 DS 89.15 / SR 70.0 (ResNet-34), NAVSIM 92.0 PDMS / 90.1 EPDMS |
| 채택 근거 | sparse query 표현 + **제안(vocabulary)+scoring 선택 구조가 본체** — 장면 요소→궤적 인과 선택(요소별 채점)을 심을 자리가 이미 있음. 폐루프 입증 + Bench2Drive 공식 브랜치(개입 데이터 수집 무대) + online mapping 유지 |
| 폴백 | TensorRT 관문(원장 §13) 실패 시 dense BEV 계열 회귀 — 되돌릴 수 있는 결정 |

## 온보딩 단계

| 단계 | 내용 | 산출물 | 상태 |
|---|---|---|---|
| **1. 자리잡기** | 슬롯·config·env 명세 (이 PR) | 이 README, `configs/sparsedrive_v2.py`, `docker/sparsedrivev2-inf/SPEC.md` | ✅ |
| 2. 재현 | 학습서버에서 upstream 그대로 공개셋 재현 (공개 가중치 → Bench2Drive/NAVSIM 점수 확인) + TensorRT 관문(껍데기 빌드·곡선) | 재현 리포트, 벤치 곡선 | — |
| 3. 반입 | 분해 온보딩: 모델 소스 → `src/`, `NOTICE.md` 고지. 공용부 승격 검토 | vendored src | — |
| 4. 수정 | 아래 "수정 4종" — 각각 별도 이슈로 | 수정 커밋 + 요소별 CG 지표 | — |
| 5. 계약 편입 | raw tier(카메라) 데이터 계약 + adapter(`policy.py`/`dataloader.py`/`postprocess.py`) + registry 등록 | 조립 계약 시민화 | — |

> **1단계 데이터 주의**: V2는 카메라 기반 — 현 데이터 계약(record parsed 테이블)에는 카메라가 없다(record에 이미지 토픽 부재 확인됨). 재현·학습은 공개셋(nuScenes full / Bench2Drive)에서 수행하고, 자체 데이터 편입은 raw tier 계약(camera REQUIRES)과 E100 동기 재수집 이후의 일이다.

## 수정 4종 (난이도 순 — 회의 결정 6~9)

1. **Goal K토큰 주입** — System 2 인터페이스: K개(8~32) 조건 토큰을 planning query attention에 추가 (원장 4.2 — 좌표 2개 아님, dual-head 이행 경로 보존)
2. **HD맵 prior 초기화** — online mapping query를 HD맵 폴리라인·signal 위치로 초기화 (맵핑을 생성→정합으로 격하). 신호등 = 맵 prior query: FOV 이탈 후에도 anchor warp로 ego 기준 위치·마지막 상태 유지
3. **요소-인식 scoring** — 기존 vocabulary-scoring에 요소별 명시 채점(신호 정지선·점유 교차·차선 위상) 추가 → "신호·차선·장애물이 이러니 이 궤적"의 인과적으로 추적 가능한 선택. 검증 지표: 요소별 반사실 갭(TL-CG, agent-ablation — 원장 §8 확장)
4. **query 생존 정책 + class-agnostic 장애물** — occlusion/FOV 이탈 관성 유지(수명·확신도 감쇠·상한) + 미지 물체의 기하 표현(언어 없음 — 기하/의미 분리 원칙)

## 재현 단계 확인 체크리스트 (2단계 진입 시)

- [ ] 공개 가중치로 Bench2Drive/NAVSIM 점수 재현 (성능 기준선 확보)
- [ ] 신호등 처리 방식 확인 (Bench2Drive는 신호 준수 채점 — 2D 헤드인지 feature인지 → 수정 2 설계 입력)
- [ ] 미관측 query 수명 처리 확인 (→ 수정 4 설계 입력)
- [ ] deformable attention TensorRT 10.3 이식성 (원장 §13 관문 — 채택이 관문을 면제하지 않음)
- [ ] Bench2Drive 브랜치의 요소 개입 API (신호 뒤집기·agent 제거 — 반사실 데이터 엔진 실현성)

## 의존 규칙 (조립 계약)

- 이 디렉토리는 `planning/interface.py` registry로만 노출된다 (5단계에서). 도메인 코드가 `src/` 내부를 직접 import하지 않는다.
- 실행 환경은 `docker/sparsedrivev2-inf/` 이미지가 소유 (`configs/sparsedrive_v2.py::env`). av-base(CPU 공용)와 독립.
