# On-Device AI 기반 실시간 얼굴 인식 시스템 (Capstone Design)

**[2024 소프트웨어종합학술대회(KSC) 포스터 세션 발표]**

## 📖 1. 프로젝트 개요

이 프로젝트는 클라우드 서버를 거치지 않고, **Raspberry Pi 4**와 같은 저사양 임베디드 디바이스에서 **실시간 얼굴 인식, 데이터 저장, 모델 학습 및 갱신**까지 모든 과정을 처리하는 '온디바이스 AI' 출입 카메라 시스템입니다.

디바이스 내부에서 독립적으로 모든 연산을 수행함으로써, 빠른 응답 속도와 데이터 프라이버시 보호라는 두 가지 목표를 달성합니다.

## ✨ 2. 주요 기능

*   **실시간 얼굴 탐지**: 웹캠 영상에서 실시간으로 얼굴 위치를 탐지합니다. (Dlib HOG/CNN, Haar Cascade)
*   **다중 모델 기반 얼굴 인식**: 정확도와 속도 등 특성이 다른 여러 모델을 구현하여 비교/선택할 수 있습니다.
    1.  **Dlib ResNet + SVM**: Dlib의 얼굴 임베딩(128d vector)과 선형 SVM을 결합한 고전적이고 빠른 방식
    2.  **Custom CNN**: 직접 설계한 간단한 CNN 모델을 이용한 방식
    3.  **MobileNetV2**: 경량 딥러닝 모델을 사용하여 높은 정확도를 추구하는 방식
*   **온디바이스 학습 및 업데이트**: 새로운 사용자가 등록되면, 디바이스가 스스로 데이터셋을 구축하고 비동기적으로 모델을 재학습하여 즉시 시스템에 반영합니다.
*   **멀티태스킹 아키텍처**: `실시간 인식`, `데이터 저장`, `모델 학습`의 무거운 작업들이 서로 방해하지 않고 동시에 실행될 수 있도록 설계되었습니다.

## 📂 3. 프로젝트 구조 및 핵심 파일

```
.
├── multitask.py             # 🔄 (메인) 인식/학습/저장 멀티태스킹 실행
├── SvmRecon.py               # 🤖 (인식) Dlib 특징 + SVM으로 실시간 인식
├── dlibRecognition.py        # 🧠 (인식) 직접 학습한 Keras CNN 모델로 실시간 인식
├── MobileRecon.py            # 📱 (인식) MobileNet 모델으로 실시간 인식
│
├── dlibTraining.py           # 🎓 (학습) SVM 또는 Custom CNN 모델 학습
├── Mobilenet.py              # 🏗️ (학습) MobileNet 모델 학습 및 구축
│
├── svm_model.joblib          # 💾 (모델) 학습된 SVM 분류기
├── my_model/                 # 📂 (모델) Keras/TensorFlow로 학습된 모델 저장 폴더
│   └── model1.keras
│   └── checkpoint_epoch_...
│
├── dataset/                  # 🖼️ (데이터) 모델 학습용 얼굴 이미지
│   ├── 0/
│   └── 1/
│
└── README.md                 # 📄 본 문서
```

## ⚙️ 4. 설치 및 실행 방법

### 의존성 설치

**중요**: `dlib`은 C++ 기반 라이브러리이므로 `cmake`가 먼저 설치되어 있어야 합니다.

```bash
# 시스템 패키지 업데이트 및 필수 도구 설치
sudo apt-get update
sudo apt-get install -y build-essential cmake
sudo apt-get install -y python3-opencv python3-pip

# Python 라이브러리 설치
pip3 install numpy dlib scikit-learn tensorflow
```

### 실행

1.  **프로젝트 클론**
    ```bash
    git clone https://github.com/kmk4729/OndeviceAI_Raspberry.git
    cd OndeviceAI_Raspberry
    ```

2.  **메인 시스템 실행**
    *   모든 기능(인식, 학습, 업데이트)을 동시에 실행합니다.
    ```bash
    python3 multitask.py
    ```

3.  **개별 인식 모델 테스트**
    *   특정 모델의 인식 기능만 독립적으로 테스트해볼 수 있습니다.
    ```bash
    python3 SvmRecon.py       # SVM 기반 인식 테스트
    # python3 dlibRecognition.py  # Custom CNN 기반 인식 테스트
    ```

## 🚀 5. 기대 효과 및 향후 개선 방향

### 기대 효과
*   **보안성 및 프라이버시 강화**: 민감한 얼굴 정보가 외부 서버로 전송되지 않습니다.
*   **실시간성 확보**: 네트워크 지연 없이 즉각적인 출입 통제가 가능합니다.
*   **확장성**: CCTV, 사원증 NFC 태그 등 기존 시스템과 쉽게 연동하여 스마트 출입 관리 시스템을 구축할 수 있습니다.

### 향후 개선 방향
*   **고성능 탐지 모델 적용**: YOLOv5/v7 등 최신 객체 탐지 모델을 경량화하여 적용.
*   **다양한 각도/자세 인식**: 측면, 후면 등 비정형적인 자세에서도 인물 인식이 가능하도록 모델 개선.
*   **통합 관제 시스템**: 서버, Web/App, DB와 연동하여 다수의 디바이스를 원격으로 관리하고, 도난 감지 등 지능형 서비스를 구축.