import cv2
import dlib
import time
# dlib 얼굴 검출기 초기화
#hog_detector = dlib.get_frontal_face_detector()  # HOG 기반 얼굴 검출기
cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")  # CNN 기반 얼굴 검출기

# 웹캠에서 비디오 스트림 읽기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
fps_start_time = time.time()
fps_counter = 0
fps_text = "FPS: calculating..."
while True:
    # 비디오 프레임 읽기
    ret, img = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fps_counter += 1


    # HOG 기반 얼굴 검출
    #faces_hog = hog_detector(frame)

    # 검출된 얼굴 주변에 초록색 사각형 그리기
    #for face in faces_hog:
    #    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # CNN 기반 얼굴 검출
    faces_cnn = cnn_detector(frame, 1)

    # 검출된 얼굴 주변에 파란색 사각형 그리기
    for face in faces_cnn:
        x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
     # FPS 계산 및 표시
        if time.time() - fps_start_time >= 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_text = f"FPS: {fps:.2f}"
            fps_counter = 0
            fps_start_time = time.time()
            
        # FPS 표시
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    # 화면에 프레임 표시
    cv2.imshow('Face Detection', img)

    # 'q' 키를 누르면 종료  12123123
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 스트림과 윈도우 닫기
cap.release()
cv2.destroyAllWindows()
