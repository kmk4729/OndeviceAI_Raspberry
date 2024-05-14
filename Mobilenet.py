import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
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
        # 이미지를 로드하고 크기를 조정합니다. (MobileNet 기본 입력 크기인 224x224로 조정)
        image = cv2.imread(image_path)
        #image = cv2.resize(image, (60, 60))
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

# MobileNet을 불러옵니다. include_top=False로 설정하여 분류기를 제외합니다.
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(60, 60, 3))
base_model.trainable = True

# 모델 구축
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(set(labels)), activation='softmax')  # 레이블 수에 맞춰 출력 노드 설정
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 데이터 증강을 위한 ImageDataGenerator 생성
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

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
model_path = 'my_model/model1_mobilenet.keras'
model.save(model_path)
