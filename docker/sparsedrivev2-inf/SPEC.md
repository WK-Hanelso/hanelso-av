# sparsedrivev2-inf — SparseDriveV2 실행 환경 명세 (자리)

System 1(SparseDriveV2) 소유 GPU 이미지. **Dockerfile은 재현 단계(온보딩 2단계)에서 실측 기반으로 작성**한다 — upstream 의존성(mmdet3d 계열 예상)을 추정으로 pin하지 않는다.

## 규약 (docker 원칙 준수)

- 이미지 = 모델 소유, `<모델>-inf` 명명. **자기 `requirements.txt` 독립** (av-base·pluto-inf 참조 금지).
- 코드 굽지 않음 — `git clone` + `docker run` 마운트 (환경/소스 분리 원칙).
- GPU 플랫폼 변형은 하나의 Dockerfile + build-arg (pluto-inf 패턴). 타깃: A6000(sm86)/5880(sm89) 학습·벤치 → Orin/Thor 배포는 별도 단계(TensorRT 10.3 관문과 함께).

## Dockerfile 작성 시 확정할 것 (재현 단계 산출)

- [ ] upstream 요구 CUDA/torch/mmcv 버전 (repo 문서 + 실빌드로 확정)
- [ ] deformable attention 커스텀 op 빌드 경로 (TensorRT 플러그인과 별개로 학습용 CUDA op)
- [ ] Bench2Drive(CARLA) 클라이언트 의존성을 이 이미지에 넣을지 분리할지
- [ ] 검증 freeze → `requirements.txt` (av-base와 동일 방식: --no-deps 재현)
