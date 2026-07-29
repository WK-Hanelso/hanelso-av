# common/ — 도메인 관통 공유 모듈

hanelso_swm의 모든 도메인(localization·perception·labeling·planning·simulation)이 공유하는 파싱 프레임워크와 데이터 포맷 정의를 담는다.

> **모듈 명세 규약**: 이 README는 `common/`의 **내용 명세**다 (코드 모듈 디렉토리당 README 1개). 설계 근거·운영 문서(FORMAT_SPEC·DESIGN·EXEC)는 별개 문서로 링크만 한다.

---

## 구성

| 경로 | 역할 |
|---|---|
| `FORMAT_SPEC.md` | 우리 통합 데이터 포맷 명세 (nuScenes 코어 + VLM 언어 레이어). **운영 문서** — 포맷 SoT. |
| `io/` | **pluggable 파싱 프레임워크** — 이종 원본을 우리 포맷으로 변환. 아래 참조. |
| `config.py` | **계층 config 조합 로더** (C-SWM-022). ROOT config(`configs/*.py`)의 모듈 이름을 `<domain>/configs/<이름>.py`로 해석·병합해 최종 config dict 하나를 반환. `_base_` 상속(dict 재귀 병합) 지원. 모든 드라이버는 root config 경로 하나만 받는다. |

## `io/` — pluggable 파싱 프레임워크

원본 종류가 늘어도 드라이버를 안 고치도록 **ABC + registry**로 구현을 갈아끼운다 ([[project_swm_architecture]], 추상화 ≤1-depth).

| 파일 | 내용 |
|---|---|
| `io/base.py` | 추상 계약 3개. `SourceParser.parse(record_path, out_dir, clip_id, pose_provider, config) -> manifest`, `EgoPoseProvider.prepare()/pose_at(ts)`, `MapParser.parse(map_path, out_dir, map_name) -> manifest`. **중간 추상 계층 없음(1-depth).** |
| `io/config.py` | `ParseConfig` dataclass. 실행 단위 입력(record/source/pose/clip_id/out_root)과 keyframe 정책(`keyframe`, `keyframe_hz`)만 담는 얇은 Python config 계약. |
| `io/registry.py` | 문자열 키 → 클래스. `register_parser/get_parser`, `register_map_parser/get_map_parser`, `register_pose/get_pose_provider`. 드라이버는 이걸로만 구현 획득(concrete 직접 import 금지). |
| `io/schema.py` | 우리 포맷 테이블(nuScenes-style) dataclass + `MapGraph` dataclass, `write_tables(out_dir, tables)`, `write_json(path, payload)` writer. 지오메트리는 python list 직렬화. |
| `io/apollo/record_parser.py` | `ApolloRecordParser(SourceParser)` — Apollo cyber record를 1-pass로 읽어 pose keyframe(기본 10Hz), ego_pose, ego_dynamics, optional obstacles→annotation/instance를 생성. import 시 registry에 `"apollo_record"` 등록. |
| `io/apollo/map_parser.py` | `ApolloMapParser(MapParser)` — Apollo `base_map.bin`을 읽어 `MapGraph`(`lanes`, `roadblocks`, `crosswalks`, `signals`, `custom_zones`)로 변환하고 `work/maps/<map_name>/map_graph.json`에 저장. import 시 registry에 `"apollo"` 등록. |
| `io/pose/identity.py` | `IdentityPoseProvider` — pose 부재 시 원점. 키 `"identity"`. |
| `io/pose/apollo_pose.py` | `ApolloRecordPoseProvider` — record `/apollo/localization/pose`(UTM+heading→quat). 키 `"apollo_record"`. |

## 확장 방법

- **새 소스 파서**: `SourceParser` 상속 → 파일 하단 `register_parser("<key>", Cls)`. 드라이버 무변경.
- **새 맵 파서**: `MapParser` 상속 → 파일 하단 `register_map_parser("<key>", Cls)`. `parse_map.py` 무변경.
- **새 pose provider**: `EgoPoseProvider` 상속 → `register_pose("<key>", Cls)`. config의 `pose="<key>"`로 선택.
- 예: E100 sensor raw(이미지+LiDAR) → `E100SensorParser`, LiDAR odometry → `LidarOdomPoseProvider`.

## 관련 문서 (운영)

- 포맷 정의: [FORMAT_SPEC.md](FORMAT_SPEC.md)
- 파서 v1 설계: [../docs/DESIGN_parser_v1.md](../docs/DESIGN_parser_v1.md)
- 실행 로그: [../agent/C-SWM-001_EXEC.md](../agent/C-SWM-001_EXEC.md)
- 드라이버: [../parse_clip.py](../parse_clip.py)
- 맵 드라이버: [../parse_map.py](../parse_map.py)
