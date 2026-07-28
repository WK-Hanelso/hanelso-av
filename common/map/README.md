# common/map

`common/map/`은 Apollo HD map 산출물(`work/maps/*/map_graph.json`)을 nuPlan `AbstractMap` 호출 표면으로 연결하는 얇은 duck-type 어댑터를 둔다.

## 구성

| 파일 | 역할 |
|---|---|
| `apollo_map.py` | `ApolloMap` + `ApolloLane` + `ApolloRoadblock` 구현. 원본 `pluto` `RouteManager`/`ScenarioManager`가 실제로 호출하는 `AbstractMap`/lane/roadblock/path 표면만 제공한다. |

## 제약

- 원본 Apollo/PLUTO/nuPlan repo 수정 없이 `map_graph.json`만 읽는다.
- pluggable 유지: 외부 호출자는 `ApolloMap(map_path=..., map_name=...)`만 알면 된다.
- 추상 계층은 `apollo_map.py` 1-depth 안에서 끝낸다.
