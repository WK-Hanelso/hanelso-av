# third_party — vendored 외부 소스

외부 repo의 소스 코드를 프로젝트 내부로 반입(vendor)하는 곳. **"git clone 하나로 재현"** 원칙을 위해, 외부 절대경로 `sys.path` 주입이나 docker 외부 마운트 대신 여기에 포함한다.

현재 이 디렉토리는 **비어 있어도 정상**이다. C-SWM-024에서 PLUTO는 해체 온보딩을 거쳐
`planning/models/pluto/src/`와 `planning/nuplan_common/`, `simulation/renderers/`로 분배됐다.

## 규약

- 이 자리는 앞으로 **진짜 외부·무수정 소스 트리**만 둔다.
- 반입 모델의 표준 절차는 vendor 고정이 아니라 **분해 온보딩**이다.
  모델의 것은 `planning/models/<이름>/src/`, 공용부는 도메인 승격(`planning/*`, `simulation/*`)으로 보낸다.
- vendor가 필요한 경우에도 **수정 금지**가 기본이다. 수정이 필요하면 모델 홈/도메인 홈으로 분해해 소유권을 옮긴다.
- 반입 시 출처·라이선스를 `<이름>/NOTICE.md`에 고지. 배포/공개 전 원저작 라이선스 포함 확인 필수.
- 라이브러리(자체 remote·버전 존재, 예: nuplan-devkit)는 vendor 대상이 아니라 docker/requirements가 설치한다 — vendor는 "remote 없는 소스 트리"만.
