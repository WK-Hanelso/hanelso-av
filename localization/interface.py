"""localization 인터페이스 placeholder (C-SWM-023 — planning/interface.py 패턴 예비).

localization 모델이 처음 들어올 때 planning/interface.py와 같은 구성으로 채운다:
- 도메인 ABC + registry (예: perception이면 Detector 등 — 실물 들어올 때 정의)
- load_model(name): import_module("localization.models.<name>") 동적 로딩 + Available 에러
지금은 구현 없음 — 추측성 일반화 금지 원칙에 따라 축만 잡아둔다.
"""
