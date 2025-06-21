# On-device AI 기반 얼굴 인식 회사 출입 카메라

## 프로젝트 개요
이 프로젝트는 클라우드 서버를 거치지 않고, Raspberry Pi 4와 같은 저사양 임베디드 디바이스에서 실시간 얼굴 인식과 학습, 모델 갱신까지 모두 처리하는 AI 기반 출입 카메라 시스템입니다.  
디바이스 내부에서 얼굴 데이터를 저장하고, 비동기적으로 학습을 진행하며, 갱신된 모델을 즉시 적용하여 실시간성과 보안성을 모두 확보합니다.

## 주요 기능
- 실시간 얼굴 인식 및 검출 (Haar Cascade, Dlib HOG, Dlib 68 Landmark)
- 신규 사용자 얼굴 데이터 저장 및 비동기 학습
- SVM 및 MobilenetV2 기반 얼굴 인식 모델 구현
- Multi-tasking 구조로 데이터 저장, 학습, 모델 갱신 동시 처리
- 리소스가 제한된 Raspberry Pi 4 환경에 최적화

## 사용 기술
- Python
- OpenCV
- Dlib
- Scikit-learn (SVM)
- TensorFlow Lite (MobilenetV2)
- Raspberry Pi 4 (쿼드코어 CPU, 4GB RAM)

## 시스템 구성 및 동작 흐름
1. **얼굴 인식**: Haar Cascade, Dlib HOG 알고리즘으로 얼굴 검출
2. **특징 추출**: Dlib 68 Landmark로 얼굴 특징점 추출
3. **데이터 저장**: 얼굴 특징 좌표를 저장하여 학습 데이터 구성
4. **학습**: SVM과 MobilenetV2 모델을 사용하여 얼굴 인식 모델 학습
5. **모델 갱신**: 비동기적으로 학습된 모델을 디바이스에 적용

## 설치 및 실행 방법
1. Raspberry Pi 4에 필요한 라이브러리 설치
    ```
    sudo apt-get update
    sudo apt-get install python3-opencv python3-pip
    pip3 install dlib scikit-learn tensorflow
    ```
2. 프로젝트 소스코드 클론
    ```
    git clone https://github.com/kmk4729/OndeviceAI_Raspberry.git
    cd OndeviceAI_Raspberry
    ```
3. 실행
    ```
    python3 multitask.py
    ```

## 기대 효과
- 외부 서버 없이도 실시간 얼굴 인식 및 출입 관리 가능
- CCTV, 사원증 NFC 태그 등 다양한 신호와 연동 가능
- 보안성 강화 및 다양한 IoT 환경에 적용 가능

## 향후 개선 방향
- YOLOv5 기반 직접 학습 가능한 영상처리 모델 개발
- 측면 및 후면 얼굴 인식 기능 추가
- 서버 및 Web/App/DB와 연동한 딥러닝 기반 실시간 도난 방지 시스템 구축

## 참고 문헌
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2019). MobileNetV2: Inverted Residuals and Linear Bottlenecks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
- King, D. E. (2009). Dlib-ml: A Machine Learning Toolkit. Journal of Machine Learning Research, 10, 1755-1758.

## 라이선스
MIT License
