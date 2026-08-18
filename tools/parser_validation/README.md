# parser_validation

`validate.py`는 Apollo cyber record 1개를 직접 읽어서 `work/maps/*/map_graph.json` 후보와 대조한 검증 리포트와 정적 BEV 시각화를 만든다. 맵 전용 검증은 record 없이 `--map-only`로 따로 실행할 수 있다.

## 계약

- 입력: `--record <path>` 또는 `--map-only <map_name>` 중 하나
- 맵 판정: `/apollo/localization/pose` UTM bbox 중심이 어느 맵의 lane bbox 안에 들어가는지 검사
- 출력:
  - `work/validation/<clip_id>/report.json`
  - `work/validation/<clip_id>/report.txt`
  - `work/validation/<clip_id>/scene_bev.png`
  - reroute가 있으면 `work/validation/<clip_id>/bev_route.png`
  - `work/maps/<map_name>/map_bev.png`

## report 내용

- `validation`: 전체 PASS/FAIL. `checks` 중 하나라도 실패하면 FAIL.
- `checks`: 동적 consistency 자동검증 결과 배열. 각 원소는 `{name, pass, detail}` 형식.
- 기본 checks:
  - `timestamp_monotonic`: pose / obstacles timestamp 단조 증가 여부
  - `displacement_vs_speed`: 인접 pose 이동거리와 chassis speed × dt 정합성
  - `agent_no_teleport`: 동일 obstacle track id의 프레임간 점프 속도 상한 검증
  - `speed_accel_bounds`: chassis speed / imu acceleration 물리 범위 검증

`report.txt`에는 `== Consistency Checks ==` 섹션이 추가되어 같은 내용을 사람이 읽기 쉬운 형태로 요약한다.

## BEV

- `map_bev.png`: map_graph 기준 전체 맵 검증 이미지. lane은 `left + reversed(right)` 도로 면 폴리곤으로 채우고, crosswalk / signal stop_line / custom_zones를 함께 표시한다.
- `scene_bev.png`: bag 기준 scene 검증 이미지. matched map 도로 면 위에 ego trajectory(속도 컬러맵, heading, S/E), 방향성 lane arrow, representative agent box, route overlay를 표시한다.
- `bev_route.png`: reroute가 있을 때만 생성되는 route 전용 비교 이미지. before / after route가 어디서 갈라지는지 확대해서 본다.
- 좌표계: `UTM52`, 축 단위: meter, `equal` aspect, 옅은 grid.

## 모듈

- `validate.py`: CLI 드라이버. 출력 디렉터리 생성, report 저장, scene/map BEV 렌더 호출 담당.
- `core.py`: record 1-pass 집계, map matching, reroute/정지구간/채널 요약, consistency checks, txt 렌더링, map/scene BEV 렌더링 담당.

## 실행

```bash
python3 tools/parser_validation/validate.py \
  --record data/bag/E100BT-25/20260716151711.record.00006
```

```bash
python3 tools/parser_validation/validate.py --map-only AYG
```

record 검증 실행 시 표준출력에는 `matched_map`, `reroute`, `validation`, `scene_bev_png`, `map_bev_png`가 함께 찍힌다.
