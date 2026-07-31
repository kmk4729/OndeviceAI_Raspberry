import cv2
import dlib
import numpy as np
import tensorflow as tf

# 모델 불러오기
model = tf.keras.models.load_model('my_model/model1.keras')

# dlib의 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()

# 웹캠에서 영상 캡처
cap = cv2.VideoCapture(0)

while True:
    # 영상 프레임 읽기
    ret, frame = cap.read()

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = detector(gray)

    # 감지된 얼굴에 대해 예측
    for face in faces:
        # 얼굴 영역 추출
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = frame[y:y+h, x:x+w]  # 컬러 이미지로 변경

        # 이미지가 비어있지 않은 경우에만 resize
        if not face_img.size == 0:
            # 얼굴 이미지를 모델의 입력 형태로 변환
            face_img = cv2.resize(face_img, (60, 60))
            face_img = np.expand_dims(face_img, axis=0)
            face_img = np.expand_dims(face_img, axis=-1)

            # 모델을 사용하여 얼굴 분류
            prediction = model.predict(face_img)

            # 분류 결과를 화면에 표시
            label = np.argmax(prediction)
            confidence = np.max(prediction)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Label: {label}, Confidence: {confidence}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 화면에 출력
    cv2.imshow('Face Recognition', frame)

    # 종료 키 확인
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
