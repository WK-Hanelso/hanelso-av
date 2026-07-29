# planning/policy

`planning/policy/`는 추론 정책을 **ABC + registry**로 갈아끼우는 최소 계층이다.

## 구성

| 파일 | 역할 |
|---|---|
| `base.py` | `Policy.infer(feature) -> dict` 추상 계약과 `register_policy/get_policy` registry. |
| `pluto_torch.py` | `PlutoTorchPolicy` 구현. hydra `model` 블록으로 `PlanningModel`을 만들고 `v3_pluto.ckpt`를 strict 로드한 뒤 eager PyTorch forward를 수행한다. |
| `__init__.py` | 외부 import 진입점. |

## 규약

- depth는 `planning/policy/*` 한 단계만 사용한다.
- 정책 구현은 입력 feature를 재정의하지 않고, 이미 준비된 batch를 그대로 consume한다.
- 새 정책은 파일 하단에서 registry 등록까지 끝내야 한다.
