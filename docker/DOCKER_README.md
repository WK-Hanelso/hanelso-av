# hanelso_swm Docker — 스펙 필요성 정리 (왜 이렇게 만들었나)

> 이 문서는 **docker 이미지가 만들어지기까지의 각 스펙 결정이 "왜 필요했는가"** 를 정리한다.
> 사용법(빌드·실행 명령)은 [README.md](README.md), 정본 명세는 [SPEC.md](SPEC.md) 참고.
> 상태: **중간 산출물(환경 이미지)**. sim + render + postprocessing 코드가 채워지면 **최종 docker**를 다시 만든다.

---

## 0. 이 이미지가 책임지는 것 / 아닌 것

목표 전체 파이프라인:
```
bag(.record) + MAP(base_map.bin)
   → parse → model input → 추론(pth/onnx, CPU) → cls 시뮬 → render → mp4
```
- **이 이미지가 책임지는 것**: 위 전 과정을 돌릴 수 있는 **"환경"(패키지·라이브러리·시스템 lib)**.
- **책임지지 않는 것**: 코드 자체. 코드는 git repo에 있고 런타임에 마운트된다. (cls 시뮬·render·onnx backend 코드는 아직 미구현 → 최종 docker 때 재검증.)

---

## 1. 왜 "환경만" 굽고 코드는 마운트하나

**필요성**: 코드는 계속 바뀐다(개발 중). 코드를 이미지에 구우면 한 줄 고칠 때마다 rebuild(수 분~십수 분)해야 한다.

**결정**: 이미지 = 환경(변화 느림), 코드 = git 마운트(변화 빠름)로 분리.
```
git clone   →  src 전부 확보 (repo의 몫)
docker build →  의존성·환경만 구성 (이미지)
docker run   →  clone된 src를 -v로 마운트해 실행
```
→ 코드 수정은 rebuild 없이 즉시 반영. deps 바뀔 때만 rebuild.

---

## 2. 왜 build.sh가 없고 `COPY <소스>`도 없나

**필요성**: "깨끗한 머신에서 `git clone` + `docker build` 만으로 동일 환경 재현"이 목표. 특정 호스트에 우연히 존재하는 로컬 파일에 의존하면 재현이 깨진다.

**결정**: Dockerfile은 `requirements.txt` 외 어떤 로컬 파일도 `COPY`하지 않는다. 로컬 소스를 임시 컨텍스트에 모으던 build.sh 방식은 이 원칙에 위배 → **폐기**.
→ Dockerfile이 필요한 것(파이썬 패키지·nuplan)은 전부 **네트워크에서 스스로 받아온다**.

---

## 3. 왜 nuplan은 Dockerfile이 설치하고, pluto_onnx는 소스로 두나

둘 다 repo 밖 의존이지만 성격이 다르다:

| | nuplan-devkit | pluto_onnx |
|---|---|---|
| 성격 | third-party **라이브러리** | 프로젝트 **소스**(PLUTO 네트워크/feature/post 코드) |
| git | 자체 remote 존재(motional) | remote 없음(그냥 소스 트리) |
| 처리 | **Dockerfile이 버전 고정 설치** `pip install --no-deps "git+…nuplan-devkit.git@e924167"` | **clone/마운트로 확보**, `run_inference --pluto-root`로 지정 |
| 이유 | 버전 고정된 외부 라이브러리라 환경의 일부 | 코드라서 clone이 가져오는 게 맞음(이미지에 굽지 않음) |

→ "환경(라이브러리)"과 "소스(코드)"의 경계를 성격에 맞춰 나눈 것.

---

## 4. 왜 CPU 전용인가 — 이식성 (핵심)

**필요성**: A6000·5880·3080·1080·2060 등 **여러 x86 머신 어디서든** 동일하게 돌아야 한다.

**문제**: GPU로 가면 GPU마다 CUDA compute capability가 달라 이미지가 특정 카드에 묶인다. (예: torch cu116 / onnxruntime-gpu는 sm_75=2060 전용이 되고 5880 Ada(sm_89)와 호환 안 됨.)

**결정**: 추론을 **CPU로 고정**(torch==1.12.0+cpu, onnxruntime=CPUExecutionProvider). PLUTO 네트워크의 natten은 코드에서 순수 torch로 대체돼(`native_nat.py`) CPU forward가 가능하다. 신경망 forward는 전체의 ~4%라 GPU 이득도 작다.

**검증된 결과**:
```
Arch = amd64 / linux
torch.cuda.is_available() = False   (torch 1.12.0+cpu)
onnxruntime providers = ['CPUExecutionProvider']
CUDA/nvidia 패키지 = 없음
```
→ **CUDA 의존 0 → x86-64 리눅스면 GPU 종류·유무와 무관하게 동작.** A6000/5880/3080/1080/2060 전부 OK.
※ CPU 유지 정책상, 만약 향후 다른 이유로 GPU가 필요해지면 GPU 이미지가 아니라 **onnx 경로로 우회**해 CPU를 유지한다.

---

## 5. 왜 torch/torchvision을 CPU판으로 "따로" 고정하나

**필요성**: `timm`(PLUTO 네트워크가 `DropPath`로 사용)의 import 체인이 torchvision을 끌어온다. 방치하면 pip가 **CUDA판 torch/torchvision**을 설치해 CPU pin이 깨진다.

**결정**: requirements보다 **먼저** `torch==1.12.0+cpu torchvision==0.13.0+cpu` 를 명시 설치(버전쌍 고정). 이후 다른 패키지가 torch를 요구해도 이미 만족돼 CUDA판이 끌려오지 않는다.

---

## 6. 왜 pip를 24.1 미만으로 낮추나

**필요성**: nuplan이 요구하는 `hydra-core==1.1.0rc1` / `omegaconf==2.1.0rc1` 은 낡은(invalid) 메타데이터(`PyYAML (>=5.1.*)` 잘못된 명세)를 가진다.

**문제**: pip 24.1+ 는 이 메타데이터를 거부 → `No matching distribution` 로 빌드 실패.

**결정**: `pip install "pip<24.1"`. 버전은 nuplan 요구대로 유지하고 pip만 낮춰 해결(pip 에러 메시지의 권장 방식).

---

## 7. 왜 requirements를 "검증 freeze 재현 + --no-deps"로 하나

**필요성**: nuplan-devkit는 의존 패키지가 매우 많다(pytest·bokeh·casadi·shapely·geopandas 등 수십 개). 최소 목록으로 시작하면 `ModuleNotFoundError`가 하나씩 터져 빌드를 수십 번 반복하게 된다(실제로 pytest 누락으로 스모크 실패 경험).

**결정**:
- 이 스택(nuplan+render+sim)의 **완전한 클로저** = 구 검증 freeze(`simulation/docker/requirements.txt`, 241개)를 베이스로 삼는다.
- 단, 그 freeze는 시스템 파이썬에서 뽑혀 **OS/apt 전용 패키지 6종**(python-apt, distro-info, unattended-upgrades, PyGObject, dbus-python, ssh-import-id)이 섞여 있어 pip 설치가 안 된다 → **제거**(235개).
- `pip install --no-deps -r requirements.txt` 로 **freeze를 그대로 재현**(resolver 미개입). 이렇게 하면 `cyber-record`가 요구하는 `protobuf<=3.19.4` vs 프로젝트의 `protobuf==3.20.3`(맵 proto 기준) 같은 상한 충돌도 회피된다.
- torch/torchvision(§5 별도), nuplan(§3 별도)만 freeze에서 빼서 따로 설치.

---

## 8. 왜 헬스체크는 "환경만" 검증하나

**필요성**: 코드(pluto_onnx/common)는 이미지에 없고 런타임 마운트된다. 그래서 빌드 시점엔 코드 import를 검증할 수 없다.

**결정**: Dockerfile의 `RUN python -c "import …"` 헬스체크는 **설치형 의존성**(torch·onnxruntime·nuplan·cv2·geopandas·hydra·timm 등)만 import해 "환경이 제대로 섰는지"를 검증한다. 코드까지 포함한 전 경로 검증은 **마운트 스모크**(run_inference)로 별도 수행한다.

**검증 결과**: 헬스체크 통과(`ENV OK | torch 1.12.0+cpu | onnxruntime 1.13.1`). 마운트 스모크에서 run_inference(pth, CPU) 출력이 호스트 baseline과 **비트 단위 동일** → 컨테이너가 호스트 결과를 정확히 재현.

---

## 9. 빌드까지의 흐름 (요약)

```
python:3.9-slim-bullseye
  → apt: ffmpeg·libgl·libsm6·libxext6·libxrender1·fonts-dejavu·git·build-essential   (§render/sim 시스템 lib)
  → pip<24.1                                                                          (§6)
  → torch==1.12.0+cpu  torchvision==0.13.0+cpu                                        (§4,5)
  → pip install --no-deps -r requirements.txt   (검증 freeze 235개)                    (§7)
  → pip install --no-deps "git+…nuplan-devkit.git@e924167"                            (§3)
  → HEALTHCHECK: 설치형 의존성 import                                                  (§8)
```
결과 이미지: `hanelso-swm-env:latest` (CPU 전용, x86-64, ~3.8GB).

---

## 10. 상태 — 최종 docker 완결 (2026-07-29)

§10의 "미완" 코드가 모두 repo에 채워졌고, **rebuild 없이** 같은 환경 위에서 컨테이너 실행이 검증됐다:
- **cls 시뮬레이션** ✅ `simulation/render_sim.py --mode closed_loop` (원본 ForwardSimulator LQR 전파 + progress-정렬 agent 재fetch)
- **render → mp4** ✅ `--renderer {matplotlib,nuplan}` (nuplan = 원본 PLUTO NuplanScenarioRender) + 컨테이너 내 ffmpeg 인코딩
- **postprocessing** ✅ `run_inference.py --postprocess` / render_sim 내부 (원본 TrajectoryEvaluator+EmergencyBrake). onnx backend는 환경만 준비(pth 사용 중).

**검증**: 컨테이너에서 run_inference(--postprocess) + render_sim(closed_loop nuplan)이 에러 없이 완주하고 postprocess 결과가 호스트와 일치, mp4 생성. 또한 **native_nat(순수 torch NAT) forward 출력이 원본 natten과 max_abs_diff ~1e-6로 수치 동등**함을 대조 검증 → CPU/natten-free 설계의 정합성 확인.

기존 requirements(freeze 235개)가 scipy·shapely·opencv·matplotlib·ffmpeg 등을 이미 포함해 **새 pip 의존 추가 없이** 동작했다.
