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
count=0
formatted_string = f"x: {x}, y: {y}"
while (webcam.isOpened()):
    ret, img = webcam.read()
    if ret == True:
        fps_counter += 1
        
        # RGB 이미지로 변환
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        # 얼굴 감지
        dets, _, _ = detector.run(gray_image, 1, 0)
        
        for det in dets:
            # 얼굴 경계 상자 그리기
            cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255, 0, 0), 2)
            land_filename = f"testdata/test2/land{count}.txt"
            image_filename = f"testdata/test2/test{count}.jpg"

            # 얼굴 랜드마크 검출
            landmarks = predictor(gray_image, det)
            # 얼굴 랜드마크 점 그리기
            with open(land_filename, 'w') as file:
                for i in range(68):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    file.write(f"{x} {y}\n")

                    cv2.circle(gray_image, (x, y), 1, (255, 255, 255), -1)           
            if cv2.waitKey(1) & 0xFF == ord('q'):

                count+=1

            cv2.imwrite(image_filename, gray_image)

        # FPS 계산 및 표시
        if time.time() - fps_start_time >= 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_text = f"FPS: {fps:.2f}"
            fps_counter = 0
            fps_start_time = time.time()
            
        # FPS 표시
        
        # 화면에 표시
        cv2.imshow("WEBCAM", gray_image)

        # 종료 조건
        if cv2.waitKey(1) == 27:
            break
        if count >= 10: # Take 30 face samples and stop video
             break

# 리소스 해제
webcam.release()
cv2.destroyAllWindows()
