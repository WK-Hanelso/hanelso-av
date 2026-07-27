# Apollo HD Map Proto — Custom Extensions Reference

> **목적**: 이 문서는 이 프로젝트(Apollo 7.0 기반, 국내 도로 대응 커스텀)에서 **stock Apollo 7.0 대비 확장된 HD Map proto**를 다른 세션/툴 개발자에게 인수인계하기 위한 코드 레벨 레퍼런스다. 이 프로젝트를 처음 보는 사람이 이 문서만으로 map 데이터 스키마와 접근 API를 이해할 수 있도록 작성됨.

---

## 0. TL;DR

- 맵의 최상위 스키마는 **`apollo.hdmap.Map`** (`modules/map/proto/map.proto`).
- `base_map.bin` = 이 `Map` 메시지를 protobuf로 직렬화한 **바이너리**. (proto2, `GetProtoFromFile`로 역직렬화)
- stock Apollo 7.0 대비 **5종의 커스텀 zone 요소가 추가**됨:
  `NoAutoDrivingZone`, `Tunnel`, `Underpass`, `ConstructionZone`, `Alleyway`.
- 이 5종은 모두 **동일 구조**: `Id id`, `repeated Id overlap_id`, `Polygon polygon` (polygon 영역 기반).
- 현재 활성 맵: `--map_dir=/apollo/modules/map/data/AYG-DNA-PCN` (`modules/common/data/global_flagfile.txt:1`). 다른 맵: `SEL-KNM2`, `SEL-MPO-SAM`, `KATRI`, `KATRI2`.

---

## 1. 최상위 스키마: `Map` 메시지

`modules/map/proto/map.proto` — package `apollo.hdmap`, `syntax="proto2"`.

```proto
message Map {
  optional Header header = 1;

  // --- stock Apollo 7.0 요소 (필드 2~13) ---
  repeated Crosswalk crosswalk = 2;
  repeated Junction junction = 3;
  repeated Lane lane = 4;
  repeated StopSign stop_sign = 5;
  repeated Signal signal = 6;
  repeated YieldSign yield = 7;
  repeated Overlap overlap = 8;
  repeated ClearArea clear_area = 9;
  repeated SpeedBump speed_bump = 10;
  repeated Road road = 11;
  repeated ParkingSpace parking_space = 12;
  repeated PNCJunction pnc_junction = 13;

  // --- 커스텀 확장 요소 (필드 14, 16~19) ---
  repeated NoAutoDrivingZone no_auto_driving_zone = 14;   // 🔧 CUSTOM
  repeated RSU rsu = 15;                                   // stock (번호가 14→15로 밀림)
  repeated Tunnel tunnel = 16;                             // 🔧 CUSTOM
  repeated Underpass underpass = 17;                       // 🔧 CUSTOM
  repeated ConstructionZone construction_zone = 18;        // 🔧 CUSTOM
  repeated Alleyway alleyway = 19;                         // 🔧 CUSTOM
}
```

> ⚠️ **stock Apollo 7.0과의 차이**: stock에서는 `Map`이 `rsu = 14`까지가 끝이다. 이 프로젝트는 `no_auto_driving_zone`을 14번에 삽입하면서 `rsu`를 15번으로 밀었고, 16~19에 나머지 커스텀 zone 4종을 추가했다. **필드 번호를 그대로 유지**해야 기존 `.bin`과 호환된다.

---

## 2. 커스텀 zone proto 5종 (전문)

**5개 모두 구조가 완전히 동일하다.** 각 파일은 `modules/map/proto/` 아래에 있다.

```proto
// modules/map/proto/map_no_auto_driving_zone.proto
syntax = "proto2";
package apollo.hdmap;
import "modules/map/proto/map_id.proto";
import "modules/map/proto/map_geometry.proto";

// A no auto driving zone means in an area which car can not driving autonomous
message NoAutoDrivingZone {
  optional Id id = 1;
  repeated Id overlap_id = 2;
  optional Polygon polygon = 3;
}
```

나머지 4종도 메시지 이름만 다르고 필드는 위와 동일하다:

| proto 파일 | 메시지 이름 | 의미 |
|---|---|---|
| `map_no_auto_driving_zone.proto` | `NoAutoDrivingZone` | 자율주행 금지구역 |
| `map_tunnel.proto` | `Tunnel` | 터널 |
| `map_underpass.proto` | `Underpass` | 지하차도 |
| `map_construction_zone.proto` | `ConstructionZone` | 공사구간 |
| `map_alleyway.proto` | `Alleyway` | 이면도로/골목 |

각 메시지 공통 필드:
```proto
message <ZoneName> {
  optional Id id = 1;              // 전역 고유 id (문자열)
  repeated Id overlap_id = 2;      // 겹치는 다른 map 요소들의 id 목록
  optional Polygon polygon = 3;    // 구역 경계 (평면 다각형)
}
```

---

## 3. 참조 기본 타입

### 3.1 `Id` — `modules/map/proto/map_id.proto`
```proto
message Id {
  optional string id = 1;   // lane/junction/overlap 등 모든 객체의 전역 고유 문자열 id
}
```

### 3.2 `Polygon` — `modules/map/proto/map_geometry.proto`
```proto
message Polygon {
  repeated apollo.common.PointENU point = 1;   // 볼록일 필요 없음(not necessary convex)
}
```

### 3.3 `PointENU` — `modules/common/proto/geometry.proto` (좌표계 핵심)
```proto
message PointENU {
  optional double x = 1 [default = nan];  // East from the origin, in meters
  optional double y = 2 [default = nan];  // North from the origin, in meters
  optional double z = 3 [default = 0.0];  // Up from WGS-84 ellipsoid, meters
}
```
> **중요**: 좌표는 **ENU 평면 좌표계(미터)** 이지 위경도가 아니다. 원점/투영은 `Map.header.projection.proj` (PROJ.4 문자열, 예: `+proj=tmerc +lat_0=... +lon_0=... +k=... +ellps=WGS84 +no_defs`)에 정의된다. 위경도↔ENU 변환이 필요하면 이 projection을 써야 한다.

### 3.4 `Header` (map.proto 내장) — 경계/투영 정보
```proto
message Header {
  optional bytes version = 1;
  optional bytes date = 2;
  optional Projection projection = 3;   // proj: PROJ.4 문자열
  optional bytes district = 4;
  ...
  optional double left = 8;    // 맵 경계 (ENU)
  optional double top = 9;
  optional double right = 10;
  optional double bottom = 11;
  optional bytes vendor = 12;
}
```

---

## 4. Overlap 연동 (요소 간 겹침 정보)

`overlap_id`가 가리키는 `Overlap` 객체는 커스텀 zone 타입도 인지한다.
`modules/map/proto/map_overlap.proto` 의 `ObjectOverlapInfo.overlap_info` oneof에 커스텀 타입이 추가돼 있다:

```proto
message NoAutoDrivingZoneOverlapInfo {}
message ConstructionZoneOverlapInfo {}
message TunnelOverlapInfo {}
message UnderpassOverlapInfo {}
message AlleywayOverlapInfo {}

message ObjectOverlapInfo {
  optional Id id = 1;
  oneof overlap_info {
    ... // stock 타입들 (lane/signal/stop_sign/...)
    NoAutoDrivingZoneOverlapInfo no_auto_driving_zone_overlap_info = 14;  // 🔧
    TunnelOverlapInfo            tunnel_overlap_info               = 15;  // 🔧
    UnderpassOverlapInfo         underpass_overlap_info            = 16;  // 🔧
    ConstructionZoneOverlapInfo  construction_zone_overlap_info    = 17;  // 🔧
    AlleywayOverlapInfo          alleyway_overlap_info             = 18;  // 🔧
  }
}
```

---

## 5. 맵 로딩 경로 (base_map.bin → Map proto)

`modules/map/hdmap/hdmap_impl.cc:58`
```cpp
int HDMapImpl::LoadMapFromFile(const std::string& map_filename) {
  Clear();
  if (absl::EndsWith(map_filename, ".xml")) {              // OpenDRIVE → Map으로 변환
    if (!adapter::OpendriveAdapter::LoadData(map_filename, &map_)) return -1;
  } else if (!cyber::common::GetProtoFromFile(map_filename, &map_)) {  // .bin/.txt 직접 역직렬화
    return -1;
  }
  return LoadMapFromProto(map_);
}
```
- `map_`의 타입은 `apollo::hdmap::Map` (`hdmap_impl.h:630` `Map map_;`, include `modules/map/proto/map.pb.h`).
- `base_map.bin`은 확장자가 `.bin`이므로 **`GetProtoFromFile`이 `Map` 스키마로 직접 파싱**한다. (파일명 탐색 순서: `base_map.bin|base_map.xml|base_map.txt`, `modules/common/configs/config_gflags.cc:28`)
- 파싱 후 `LoadMapFromProto`가 각 요소를 KD-Tree/해시테이블로 인덱싱한다.

---

## 6. C++ 소비 API (HDMap public interface)

커스텀 zone은 stock 요소와 동일한 패턴의 조회 API를 갖는다. (`modules/map/hdmap/hdmap.h`)

### 6.1 ID로 단건 조회 (`hdmap.h:81-85`)
```cpp
NoAutoDrivingZoneInfoConstPtr GetNoAutoDrivingZoneById(const Id& id) const;
ConstructionZoneInfoConstPtr  GetConstructionZoneById(const Id& id) const;
TunnelInfoConstPtr            GetTunnelById(const Id& id) const;
UnderpassInfoConstPtr         GetUnderpassById(const Id& id) const;
AlleywayInfoConstPtr          GetAlleywayById(const Id& id) const;
```

### 6.2 반경 내 조회 (`hdmap.h:163-200`)
```cpp
int GetNoAutoDrivingZones(const PointENU& point, double distance,
                          std::vector<NoAutoDrivingZoneInfoConstPtr>* out) const;
int GetConstructionZones(const PointENU& point, double distance,
                         std::vector<ConstructionZoneInfoConstPtr>* out) const;
int GetTunnels     (const PointENU& point, double distance, std::vector<TunnelInfoConstPtr>*     out) const;
int GetUnderpasses (const PointENU& point, double distance, std::vector<UnderpassInfoConstPtr>*  out) const;
int GetAlleyways   (const PointENU& point, double distance, std::vector<AlleywayInfoConstPtr>*   out) const;
```

### 6.3 Info 래퍼 클래스 (`hdmap_common.h:477-575`)
5종 모두 동일 패턴. 예:
```cpp
class NoAutoDrivingZoneInfo {
 public:
  explicit NoAutoDrivingZoneInfo(const NoAutoDrivingZone& z);
  const Id& id() const;                                     // 원본 id
  const NoAutoDrivingZone& no_auto_driving_zone() const;    // 원본 proto 접근
  const apollo::common::math::Polygon2d& polygon() const;   // 계산된 2D 폴리곤 (기하 연산용)
 private:
  const NoAutoDrivingZone& no_auto_driving_zone_;
  apollo::common::math::Polygon2d polygon_;
};
```
- proto의 `Polygon`(PointENU 목록)이 로딩 시 `apollo::common::math::Polygon2d`로 변환되어 점-포함/거리 등 기하 연산에 쓰인다.
- 각 zone은 `*PolygonKDTree`(AABoxKDTree2d)로 인덱싱되어 반경 조회가 빠르다.
- shared_ptr alias: `TunnelInfoConstPtr = std::shared_ptr<const TunnelInfo>` 등 (`hdmap_common.h:142-146`).

---

## 7. Planning 모듈에서의 실제 소비 (다운스트림 예시)

- **Traffic rule**: `modules/planning/traffic_rules/no_auto_driving_zone.{h,cc}` — planning이 매 사이클 이 zone을 읽어 자율주행 금지구역 판정에 사용.
- **Learning data**: `modules/planning/common/message_process.cc` 가 `GetNoAutoDrivingZones`/`GetConstructionZones`/`GetTunnels`/`GetUnderpasses`/`GetAlleyways` 를 호출.
- **출력 반영**: planning 결과(`ADCTrajectory`)에 `no_auto_driving_zone_type`, `construction_zone_type` 필드가 실려 `CanEdgeMonitor`로 발행됨 (`modules/planning/planning_component.cc:474-479`).

즉 **map proto 확장 → HDMap이 로딩/인덱싱 → planning traffic rule이 조회 → 주행구역 판정/출력**의 국내(한국) 도로 대응 파이프라인.

---

## 8. 툴 개발 시 체크리스트

1. **파싱**: `base_map.bin`을 읽으려면 `apollo.hdmap.Map` 스키마(위 map.proto + 5개 커스텀 proto import)로 컴파일된 protobuf가 필요하다. stock Apollo 7.0 proto만 쓰면 **필드 14·16~19를 unknown field로 흘려버린다** (파싱은 되지만 커스텀 zone을 못 읽음).
2. **좌표**: 모든 point는 ENU 미터. 위경도가 필요하면 `Map.header.projection.proj`(PROJ.4)로 역투영.
3. **id 관계**: zone의 `overlap_id` → `Map.overlap[]`의 `Overlap.id`와 매칭 → 어떤 lane/junction과 겹치는지 추적 가능.
4. **필드 번호 고정**: 스키마를 수정하더라도 14/15/16/17/18/19 번호를 바꾸면 기존 `.bin`과 깨진다.
5. **정본(定本) 위치**: 항상 `modules/map/proto/*.proto`를 소스 오브 트루스로 삼을 것. 이 문서는 요약일 뿐 수치/필드는 실제 proto로 재확인.

---

## 부록 A. 관련 파일 인덱스

| 목적 | 경로 |
|---|---|
| 최상위 스키마 | `modules/map/proto/map.proto` (`Map` msg) |
| 커스텀 zone proto | `modules/map/proto/map_{no_auto_driving_zone,tunnel,underpass,construction_zone,alleyway}.proto` |
| 기본 타입 | `modules/map/proto/map_id.proto`, `map_geometry.proto`, `modules/common/proto/geometry.proto` |
| Overlap 타입 | `modules/map/proto/map_overlap.proto` |
| 로딩 구현 | `modules/map/hdmap/hdmap_impl.cc:58` (`LoadMapFromFile`), `hdmap_impl.h:630` |
| 공개 API | `modules/map/hdmap/hdmap.h:81-85, 163-200` |
| Info 래퍼 | `modules/map/hdmap/hdmap_common.h:477-575` |
| 활성 맵 지정 | `modules/common/data/global_flagfile.txt:1` (`--map_dir`) |
| 맵 데이터 | `modules/map/data/{AYG-DNA-PCN,SEL-KNM2,SEL-MPO-SAM,KATRI,KATRI2}/base_map.bin` |
| Planning 소비 | `modules/planning/traffic_rules/no_auto_driving_zone.cc`, `modules/planning/common/message_process.cc` |
