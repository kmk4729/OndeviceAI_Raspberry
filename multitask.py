import cv2
import dlib
import logging

import numpy as np
import tensorflow as tf
import threading
import os
import queue
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # GPU 메모리 할당 비율 조정
config.gpu_options.allow_growth = True  # GPU 메모리 자동 증가 설정
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_epsilon(session)  # Keras에 TensorFlow 세션 설정

# 학습 로그는 레벨 3로 설정하여 표시합니다.
tf.get_logger().setLevel('INFO')
# 함수 정의: 얼굴 인식 및 인식 결과를 화면에 표시하는 작업  
def get_user_name_from_count_txt(index):
    with open('count.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(':')
            if parts[0] == str(index):
                return parts[1]
    return "Unknown"

def update_image_count(txt_file, user_id, user_name, count):
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            lines = file.readlines()
        found = False
        with open(txt_file, 'w') as file:
            for line in lines:
                if line.startswith(user_id):
                    file.write(f"{user_id}:{user_name}:{count}\n")
                    found = True
                else:
                    file.write(line)
        if not found:
            with open(txt_file, 'a') as file:
                file.write(f"{user_id}:{user_name}:{count}\n")
    else:
        with open(txt_file, 'w') as file:
            file.write(f"{user_id}:{user_name}:{count}\n")


def face_recognition():
    capture_switch=False

    # 모델 불러오기
    model = tf.keras.models.load_model('my_model/model1.keras')

    # dlib의 얼굴 감지기 초기화
    detector = dlib.get_frontal_face_detector()

    # 웹캠에서 영상 캡처
    cap = cv2.VideoCapture(0)
    cap.set(3, 320) # set video width
    cap.set(4, 240)

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

                # softmax 활성화 함수를 사용하여 출력을 확률 분포로 변환
                probabilities = tf.nn.softmax(prediction[0]).numpy()

                # 가장 높은 확률을 가진 클래스의 인덱스를 가져옴
                label = np.argmax(probabilities)
                
                # 가장 높은 확률이 50% 미만인 경우
                #user_id = input("Enter user ID: ")  # 사용자 ID 입력 받음
                
                # 해당 user_id에 해당하는 폴더에 30장의 사진 저장
                if capture_switch==False:
                    thread_capture = threading.Thread(target=input_key,args=(cap,detector,sw))
                    thread_capture.start()
                    capture_switch=True
                #capture_images(user_id,cap,i)
                
                # 멀티스레딩으로 training 시작
                if user_id_queue.qsize()==0:
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
                            # softmax 활성화 함수를 사용하여 출력을 확률 분포로 변환
                            probabilities = tf.nn.softmax(prediction[0]).numpy()

                            # 가장 높은 확률을 가진 클래스의 인덱스를 가져옴
                            label = np.argmax(probabilities)
                            
                            # 가장 높은 확률이 50% 미만인 경우
                            confidence = probabilities[label] * 100  # 백분율로 변환
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            user_name = get_user_name_from_count_txt(label)

                            cv2.putText(frame, f'{user_name}, Confidence: {confidence:.2f}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # 화면에 출력
                    cv2.imshow('Face Recognition', frame)

                    # 종료 키 확인
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    capture_images(cap,detector,sw)
                    thread_train_model = threading.Thread(target=train_model)
                    thread_train_model.start()
                    # train_model 스레드가 종료될 때까지 얼굴 감지 및 인식 반복
                    while thread_train_model.is_alive():  
                        # 웹캠에서 영상 캡처
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

                                    # softmax 활성화 함수를 사용하여 출력을 확률 분포로 변환
                                    probabilities = tf.nn.softmax(prediction[0]).numpy()

                                    # 가장 높은 확률을 가진 클래스의 인덱스를 가져옴
                                    label = np.argmax(probabilities)
                                    
                                    # 가장 높은 확률이 50% 미만인 경우
                                    confidence = probabilities[label] * 100  # 백분율로 변환
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                    user_name = get_user_name_from_count_txt(label)

                                    cv2.putText(frame, f'{user_name}, Confidence: {confidence:.2f}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                            # 화면에 출력
                            cv2.imshow('Face Recognition', frame)

                            # 종료 키 확인
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                    

                    # train_model 스레드가 종료된 후에 join()
                    thread_train_model.join()  # training 스레드 종료 대기
                    detector = dlib.get_frontal_face_detector()
                    model = tf.keras.models.load_model('my_model/model1.keras')

                    thread_capture.join()
                    capture_switch=False

                    # 얼굴 인식 중지
                        
                # 분류 결과와 해당 클래스의 확률을 화면에 표시
                confidence = probabilities[label] * 100  # 백분율로 변환
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # 화면에 출력
        cv2.imshow('Face Recognition', frame)

        # 종료 키 확인
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료
    cap.release()
    cv2.destroyAllWindows()

def input_key(cap,detector,sw):
    sw=False
    id = input("Enter user ID: ")
    user_id_queue.put(id)
    sw=True
    
def capture_images(cap, detector,sw):
    # Initialize HOG face detector from dlib
    # face_detector = dlib.get_frontal_face_detector()
    user_id = user_id_queue.get()
    print(user_id)
    # Initialize landmark predictor
    print(f"\n [INFO] Initializing face capture for user ID {user_id}. Look at the camera and wait ...")

    # Load or initialize image count from text file
    count_file_path = "count.txt"
    if os.path.exists(count_file_path):
        with open(count_file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith(user_id):
                count = int(line.split(":")[2])
                user_name = line.split(":")[1]
                break
        else:
            count = 0
            user_name = input(f"Enter user name for ID {user_id}: ")
            update_image_count(count_file_path, user_id, user_name, count)
    else:
        count = 0
        user_name = input(f"Enter user name for ID {user_id}: ")
        update_image_count(count_file_path, user_id, user_name, count)

    newcount = 0
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #etector2 = dlib.get_frontal_face_detector()


        # Detect faces using HOG face detector
        faces = detector(gray)
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            formatted_string = f"x: {w}, y: {h}"
            cv2.putText(img, formatted_string, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            count += 1
            newcount += 1
            
            # Save the captured image into the datasets folder
            face_img = gray[y:y+h, x:x+w]
            if len(face_img) > 0:
                face_img_normalized = face_img / 255.0
                # Resize image to 60x60
                if not face_img_normalized.size == 0:
                    face_img_resized = cv2.resize(face_img_normalized, (60, 60))
                    face_img_resized *= 255
                    image_filename = f"dataset/{user_id}/User_{user_id}_{count}.jpg"
                    os.makedirs(os.path.dirname(image_filename), exist_ok=True)
                    cv2.imwrite(image_filename, face_img_resized)
                    # Detect landmarks
                    # Save landmarks to fil
                

                # Detect landmarks
                # Save landmarks to file

            #cv2.imshow('Face Recognition', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif newcount >= 10: # Take 30 face samples and stop video
             break

    # Update or create text file with updated image count and user name
    update_image_count(count_file_path, user_id, user_name, count)

    # Do a bit of cleanup
    print(f"\n [INFO] Exiting face capture for user ID {user_id} and cleaning up")
    


# 함수 정의: 모델 학습
# 함수 정의: 모델 학습
def train_model():
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
    batch_size = 64
    augmented_train_generator = datagen.flow(train_images, train_labels, batch_size=batch_size)

    history = model.fit(
        augmented_train_generator,
        steps_per_epoch=train_images.shape[0] // batch_size,
        epochs=1,
        validation_data=(test_images, test_labels)
    )

    # 모델 평가
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print("Test Accuracy:", test_accuracy)

    # 모델 저장
    model_path = 'my_model/model1.keras'
    model.save(model_path)

# 함수 정의: 해당 user_id에 해당하는 폴더에 30장의 사진 저장


# 멀티스레딩 시작
sw=False
thread_face_recognition = threading.Thread(target=face_recognition)
user_id_queue = queue.Queue()

# 얼굴 인식 스레드 시작
thread_face_recognition.start()

# 얼굴 인식 스레드 종료 대기
thread_face_recognition.join()
