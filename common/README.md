# common/ — 도메인 관통 공유 모듈

hanelso_swm의 모든 도메인이 공유하는 **순수 공유물**만 남긴다. 파싱 프레임워크(구
`common/io`)는 C-SWM-023에서 [`data_devkit/`](../data_devkit/README.md)로 승격했다.

> **모듈 명세 규약**: 이 README는 `common/`의 **내용 명세**다 (코드 모듈 디렉토리당 README 1개). 설계 근거·운영 문서(FORMAT_SPEC·DESIGN·EXEC)는 별개 문서로 링크만 한다.

## 구성

| 경로 | 역할 |
|---|---|
| `FORMAT_SPEC.md` | 우리 통합 데이터 포맷 명세 (nuScenes 코어 + VLM 언어 레이어). **운영 문서** — 포맷 SoT. |
| `config.py` | **계층 config 조합 로더** (C-SWM-022). ROOT config(`configs/*.py`)의 모듈 이름을 `<domain>/configs/<이름>.py`로 해석·병합해 최종 config dict 하나를 반환. `_base_` 상속(dict 재귀 병합) 지원. 모든 드라이버는 root config 경로 하나만 받는다. |

## 사용 예

```python
from common.config import load_config
cfg = load_config("configs/e100bt25.py")
cfg["planning"]["policy"]   # -> "pluto_torch"
```

## 관련 문서 (운영)

- 포맷 정의: [FORMAT_SPEC.md](FORMAT_SPEC.md)
- 파싱·데이터 계약: [../data_devkit/README.md](../data_devkit/README.md)
- 실행 로그: [../agent/C-SWM-001_EXEC.md](../agent/C-SWM-001_EXEC.md)
