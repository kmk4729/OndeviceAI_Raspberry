import dlib
import cv2
import time
import os
import numpy as np
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

# FPS 관련 변수 초기화
fps_start_time = time.time()
fps_counter = 0
fps_text = "FPS: calculating..."
x=0
y=0
count=0
formatted_string = f"x: {x}, y: {y}"
while (webcam.isOpened()):
    ret, img = webcam.read()
    if ret == True:
        fps_counter += 1
        
        # RGB 이미지로 변환
        #gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        # 얼굴 감지
        dets, _, _ = detector.run(img, 1, 0)
        
        for det in dets:
            # 얼굴 경계 상자 그리기
            land2=[]
            #cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255, 0, 0), 2)
            image_filename = f"224test/2/test{count}.jpg"

            # 얼굴 랜드마크 검출
            landmarks = predictor(img, det)
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
            img_homo = cv2.warpPerspective(img, H_inv, (w, h))
            img_slice = img_homo[87:202,101:210]
            face_img_resized = cv2.resize(img_slice, (224, 224))


            count+=1

            cv2.imwrite(image_filename, face_img_resized)
            print(f"{count} th picture is saved..")
            
        # FPS 표시
        
        # 화면에 표시
        cv2.imshow("WEBCAM", img)

        # 종료 조건
        if cv2.waitKey(1) == 27:
            break
        if count >= 1000: # Take 30 face samples and stop video
             break

# 리소스 해제
webcam.release()
cv2.destroyAllWindows()
