
import tensorflow as tf
# 모델 불러오기
import dlib
import cv2
import time
import os
import numpy as np
from joblib import load

# 저장된 SVM 모델 로드
svm_model = load('svm_model.joblib')
# dlib의 얼굴 검출기 생성
detector = dlib.get_frontal_face_detector()
# 얼굴 랜드마크 검출기 생성
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
file_path1=f"land0.txt"
land1=[]
land2=[]
if os.path.exists(file_path1):
            with open(file_path1, 'r') as file1:
                lines1 = file1.readlines()
            for line1 in lines1:
                land1.append((int(line1.split(" ")[0]),int(line1.split(" ")[1])))
numland1 = np.array(land1, dtype=np.float32)

# 웹캠 열기
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


# dlib의 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()

# 웹캠에서 영상 캡처
cap = cv2.VideoCapture(0)

while (webcam.isOpened()):
    ret, img = webcam.read()
    if ret == True:
        
        # RGB 이미지로 변환
        #gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2=img
        
        # 얼굴 감지
        dets, _, _ = detector.run(img2, 1, 0)
        
        for det in dets:
            # 얼굴 경계 상자 그리기
            land2=[]
            cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255, 0, 0), 2)

            # 얼굴 랜드마크 검출
            landmarks = predictor(img2, det)
            # 얼굴 랜드마크 점 그리기
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                land2.append((x,y))
                #file.write(f"{x} {y}\n")
                #cv2.circle(gray_image, (x, y), 1, (255, 255, 255), -1)
                             
            numland2 = np.array(land2, dtype=np.float32)
            retval, mask = cv2.findHomography(numland1, numland2, cv2.RANSAC)
            h, w = 240,320
            #print(f"max : {maxlandx}, may : {maxlandy}, mix : {minlandx}, miy : {minlandy}")
            H_inv =  np.linalg.inv(retval)
            img_homo = cv2.warpPerspective(img2, H_inv, (w, h))
            face_img = img_homo[87:202,101:210]
            if not face_img.size == 0:
            # 얼굴 이미지를 모델의 입력 형태로 변환
                face_img = cv2.resize(face_img, (224, 224))
                face_img = np.expand_dims(face_img, axis=0)

                # 모델을 사용하여 얼굴 분류
                prediction = model.predict(face_img)

                # SVM 모델을 사용하여 얼굴 추가 인식
                svm_prediction = svm_model.predict(prediction)

                # 분류 결과를 화면에 표시
                label = np.argmax(svm_prediction)
                confidence = np.max(svm_prediction)
                cv2.putText(img, f'Label: {label}, Confidence: {confidence}', (det.left(), det.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# FPS 표시
        
        # 화면에 표시
        cv2.imshow("WEBCAM", img)

        # 종료 조건
        if cv2.waitKey(1) == 27:
            break


cap.release()
cv2.destroyAllWindows()