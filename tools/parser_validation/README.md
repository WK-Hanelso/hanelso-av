# parser_validation

`validate.py`는 Apollo cyber record 1개를 직접 읽어서 `work/maps/*/map_graph.json` 후보와 대조한 검증 리포트를 만든다.

## 계약

- 입력: `--record <path>`
- 맵 판정: `/apollo/localization/pose` UTM bbox 중심이 어느 맵의 lane bbox 안에 들어가는지 검사
- 출력: `work/validation/<clip_id>/report.json`, `report.txt`

## 모듈

- `validate.py`: CLI 드라이버. 출력 디렉터리 생성과 파일 저장만 담당.
- `core.py`: record 1-pass 집계, map matching, reroute/정지구간/채널 요약, txt 렌더링 담당.

## 실행

```bash
.venv-apollo/bin/python tools/parser_validation/validate.py \
  --record data/bag/E100BT-25/20260716151711.record.00006
```
