# tools — 횡단 지원 도구

특정 도메인에 속하지 않는 시각화·검증·검수 도구 모음 (CHARTER §2의 tools 도메인).

## 구조

| 디렉토리 | 역할 |
|---|---|
| `parser_validation/` | 파싱 결과 검증 baseline — 맵 자동판정, 채널 요약, 물리 정합 checks(4종), 맵/scene BEV, reroute 검출 |

## 사용 예

```bash
docker run --rm -v "$PWD":/workspace -w /workspace swm-base:latest \
  -v /mnt/hdd_storage:/mnt/hdd_storage \
  python tools/parser_validation/validate.py \
    --record data/bag/E100BT-25/20260716151711.record.00006
# → work/validation/<clip>/{report.json, report.txt, scene_bev.png}
```

## 규약

- 도구는 `data_devkit`/도메인 산출물을 **읽기만** 한다(원본·파생물 수정 금지).
- 도메인에 종속되는 로직이 생기면 해당 도메인으로 이동 검토 (예: parser_validation의 계약 검사 → data_devkit 통합 검토 중).
