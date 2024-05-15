import cv2
import dlib
import numpy as np
from joblib import load

# SVM 모델 로드
svm_model = load('224svm_model.joblib')

# dlib의 얼굴 검출기 생성
detector = dlib.get_frontal_face_detector()
# 얼굴 랜드마크 검출기 생성
predictor = dlib.shape_predictor('my_model/shape_predictor_68_face_landmarks.dat')
# 얼굴 인식 모델 생성
facerec = dlib.face_recognition_model_v1('my_model/dlib_face_recognition_resnet_model_v1.dat')

def find_faces(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR 이미지를 RGB로 변환
    dets = detector(img_rgb, 1)
    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.float64)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = predictor(img_rgb, d)
        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
    return rects, shapes, shapes_np

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)

# 웹캠 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 얼굴 검출
    rects, shapes, _ = find_faces(frame)
    
    for shape in shapes:
        # 얼굴 특징 벡터 추출
        face_descriptor = encode_faces(frame, [shape])[0]
        label = svm_model.predict([face_descriptor])[0]
        proba = svm_model.decision_function([face_descriptor])[0]
        confidence = np.max(proba)
        # SVM 모델을 사용하여 얼굴 인식
        
        
        # 화면에 얼굴 주변에 박스와 인식된 레이블 및 정확도 표시
        cv2.rectangle(frame, (shape.rect.left(), shape.rect.top()), (shape.rect.right(), shape.rect.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, f'Label: {label}, Confidence: {confidence:.2f}', (shape.rect.left(), shape.rect.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 화면에 표시
    cv2.imshow('Face Recognition', frame)
    
    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 종료 및 창 닫기
cap.release()
cv2.destroyAllWindows()
