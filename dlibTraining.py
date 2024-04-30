import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# 데이터 증강을 위한 ImageDataGenerator 생성
datagen = ImageDataGenerator(
    rotation_range=20,      # 회전 각도 범위 (0~180)
    width_shift_range=0.1,  # 가로 방향 이동 범위 (전체 너비의 비율)
    height_shift_range=0.1, # 세로 방향 이동 범위 (전체 높이의 비율)
    shear_range=0.2,        # 전단 강도 범위
    zoom_range=0.2,         # 확대/축소 범위
    horizontal_flip=True,   # 수평 뒤집기 여부
    fill_mode='nearest'     # 이미지를 회전하거나 이동할 때 채울 픽셀 전략
)

# 모델 구축
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

# 데이터 증강 후 모델 학습
batch_size = 32
augmented_train_generator = datagen.flow(train_images, train_labels, batch_size=batch_size)

history = model.fit(
    augmented_train_generator,
    steps_per_epoch=train_images.shape[0] // batch_size,
    epochs=20,
    validation_data=(test_images, test_labels)
)

# 모델 평가
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)

# 모델 저장
model_path = 'my_model/model1.keras'
model.save(model_path)
