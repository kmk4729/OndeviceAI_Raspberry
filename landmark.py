import dlib
import cv2
import time

# dlib의 얼굴 검출기 생성
detector = dlib.get_frontal_face_detector()
# 얼굴 랜드마크 검출기 생성
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
formatted_string = f"x: {x}, y: {y}"
while (webcam.isOpened()):
    ret, img = webcam.read()
    if ret == True:
        fps_counter += 1
        
        # RGB 이미지로 변환
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        # 얼굴 감지
        dets, _, _ = detector.run(rgb_image, 1, 0)
        
        for det in dets:
            # 얼굴 경계 상자 그리기
            #cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255, 0, 0), 2)
            
            # 얼굴 랜드마크 검출
            landmarks = predictor(rgb_image, det)
            bigx=-1000
            bigy=-1000
            smallx=1000
            smally=1000
            # 얼굴 랜드마크 점 그리기
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                bigx=max(x,bigx)
                bigy=max(y,bigy)
                smallx=min(x,smallx)
                smally=min(y,smally)
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            cv2.rectangle(img, (bigx, bigy), (smallx, smally), (255, 0, 0), 2)
            x=bigx-smallx
            y=bigy-smally
            formatted_string = f"x: {x}, y: {y}"

            cv2.putText(img, formatted_string, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # FPS 계산 및 표시
        if time.time() - fps_start_time >= 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_text = f"FPS: {fps:.2f}"
            fps_counter = 0
            fps_start_time = time.time()
            
        # FPS 표시
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 화면에 표시
        cv2.imshow("WEBCAM", img)

        # 종료 조건
        if cv2.waitKey(1) == 27:
            break

# 리소스 해제
webcam.release()
cv2.destroyAllWindows()
