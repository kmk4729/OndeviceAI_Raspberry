import dlib
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

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

# 이미지가 있는 폴더 경로
data_dir = "224test"
# 이미지와 레이블을 저장할 리스트 초기화
descriptor_list = []
labels = []
images = []
# 각 폴더에 있는 이미지를 로드하고 레이블을 설정합니다.
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    for image_name in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_name)
        # 이미지를 로드하고 크기를 조정합니다.
        image = cv2.imread(image_path)
        image = cv2.resize(image, (60, 60))
        # 이미지 데이터를 리스트에 추가합니다.
        images.append(image)
        rects, shapes, _ = find_faces(image)
        descriptors = encode_faces(image, shapes)
        if(len(descriptors)!=0):
            descriptor_list.append(descriptors[0])
            # 레이블을 리스트에 추가합니다.
            labels.append(int(label))

# 이미지와 레이블을 NumPy 배열로 변환합니다.
images = np.array(images)
labels = np.array(labels)

# 얼굴을 인식하고 특징 벡터를 추출합니다.
# 여기서는 하나의 얼굴만 고려합니다.
descriptor_list = np.array(descriptor_list)

# 이미지와 레이블을 저장할 리스트 초기화


# 데이터를 학습 및 테스트 세트로 분할합니다.
train_features, test_features, train_labels, test_labels = train_test_split(descriptor_list, labels, test_size=0.2, random_state=42)

# SVM 분류기 정의 및 학습
svm_model = make_pipeline(StandardScaler(), SVC())
svm_model.fit(train_features, train_labels)

# 모델 저장
svm_model_path = '224svm_model.joblib'
dump(svm_model, svm_model_path)
