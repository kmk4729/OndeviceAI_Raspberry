import cv2
import time

# 웹캠 연결
cap = cv2.VideoCapture(0)

# FPS 계산을 위한 변수 초기화
fps_start_time = time.time()
fps_counter = 0
fps = fps_counter / (time.time() - fps_start_time)
fps_text = f"FPS: {fps:.2f}"
while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # FPS 측정 및 표시
    fps_counter += 1
    if time.time() - fps_start_time >= 1:  # 1초마다 FPS 업데이트
        fps = fps_counter / (time.time() - fps_start_time)
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        fps_counter = 0
        fps_start_time = time.time()

    # 화면에 표시
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Webcam', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()