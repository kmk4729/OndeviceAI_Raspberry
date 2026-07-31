# 진행 상황 & 할 일 정리

## ✅ 완료된 것 (구조 정리)

- 스크립트를 `train/`(PC 학습) / `deploy/`(Pi 실행) / `utils/`(검출·랜드마크·벤치마크 보조) / `custom_model/`(직접 설계용, 빈 폴더)로 재구성
- `dataset/`, `my_model/`, `224test/`, `testdata/`, `*.joblib`, `*.dat`, 상태 파일(`count.txt` 등)을 `.gitignore` 처리 (로컬 디스크엔 그대로 있음, git 추적만 해제)
- `requirements-pc.txt`(학습용, full tensorflow) / `requirements-pi.txt`(배포용, tflite-runtime) 분리
- README 병합 충돌 잔재 정리, 새 폴더 구조에 맞게 갱신

> ⚠️ 위 변경사항은 아직 **커밋 전(스테이징만 된 상태)**. 확인 후 커밋 필요.

## 🧠 배포 아키텍처에 대한 결론 (기억용 메모)

- **모델 파일**(`.tflite`, `.onnx`)은 아키텍처 종속적이지 않음 → PC(x86)에서 만든 파일을 그대로 Pi(ARM)로 복사해서 씀. 크로스 컴파일 불필요.
- 크로스 컴파일/빌드 시간이 드는 건 **네이티브 런타임 라이브러리**뿐 (`tensorflow`/`tflite-runtime`은 prebuilt wheel 있음, `dlib`은 Pi에서 소스 빌드될 수 있어 piwheels 사용 권장).
- ARM/AMD를 하나로 통합하는 배포 방법 = **TFLite(또는 ONNX) 같은 이식 가능한 중간 포맷 사용** — 이미 이 방향으로 가고 있음(`train/tflite.py`).

## 🧩 지금 해야 할 일

### 1. 모델 설계 (`custom_model/`)
- [ ] 직접 설계할 모델 구조 결정 (MobileNetV2 frozen backbone + 얇은 classifier head 방식 참고)
- [ ] 표준 Conv/DepthwiseConv/Dense 위주로 설계 → 이후 int8 양자화가 잘 먹히도록
- [ ] 신규 유저 등록 시 **backbone은 그대로 두고 head만 재학습**하는 구조로 설계 (현재 `deploy/multitask.py`처럼 전체 CNN을 매번 처음부터 재학습하지 않도록)

### 2. 양자화 파이프라인
- [ ] `train/tflite.py`에 representative dataset 추가해서 **full-integer(int8) 양자화**로 확장
- [ ] 양자화 전/후 Pi에서 실제 인식 정확도 비교 측정 (손실 크면 QAT 검토)

### 3. Pi 배포 검증
- [ ] `requirements-pi.txt`로 실제 Pi에 설치 테스트 (`dlib`이 piwheels에서 prebuilt로 잡히는지 확인, 안 되면 빌드 시간 감안)
- [ ] 새 폴더 구조에서 `python3 deploy/multitask.py`가 **프로젝트 루트 기준**으로 정상 동작하는지 확인

### 4. 기존 코드 개선 (여유 있을 때)
- [ ] `deploy/multitask.py`의 전처리 불일치 정리 (`/255.0` 정규화 후 저장 직전에 다시 `*255` 곱해서 정규화가 무의미해지는 부분)
- [ ] 멀티태스킹을 `threading` → `multiprocessing`으로 전환 검토 (GIL 때문에 CPU-bound 작업이 실제로는 병렬로 안 돌 가능성)
- [ ] 매 유저 등록마다 전체 데이터셋 재학습하는 구조를 backbone-freeze 방식으로 교체

### 5. git 정리 (판단 필요, 아직 보류)
- [ ] 스테이징된 `.gitignore`/폴더 이동 변경사항 커밋 여부 결정
- [ ] `.git` 히스토리 자체 용량(384MB)까지 줄일지 결정 — 줄이려면 `git filter-repo`/BFG로 과거 커밋 재작성 필요(파괴적, 커밋 해시 변경됨)
