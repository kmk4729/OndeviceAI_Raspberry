import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# 이미지가 있는 폴더 경로
data_dir = "dataset"

# 이미지와 레이블을 저장할 리스트 초기화
images = []
labels = []

# 각 폴더에 있는 이미지를 로드하고 레이블을 설정합니다.
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    for image_name in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_name)
        # 이미지를 로드하고 크기를 조정합니다. (예: 100x100)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (60, 60))
        # 이미지 데이터를 리스트에 추가합니다.
        images.append(image)
        # 레이블을 리스트에 추가합니다.
        labels.append(int(label))

# 이미지와 레이블을 NumPy 배열로 변환합니다.
images = np.array(images)
labels = np.array(labels)
images = images / 255.0
# 데이터를 학습 및 테스트 세트로 분할합니다.
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
# 이미지 데이터를 0과 1 사이의 값으로 정규화


# CNN 모델 구축
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(set(labels)), activation='softmax')  # 레이블 수에 맞춰 출력 노드 설정
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels))

# 모델 평가
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)
model_path = 'my_model/model1.keras'

# 모델 저장
model.save(model_path)