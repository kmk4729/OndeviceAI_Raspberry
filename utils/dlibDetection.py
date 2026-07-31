import dlib
import cv2
import time

detector = dlib.get_frontal_face_detector()
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
fps_start_time = time.time()
fps_counter = 0
fps = fps_counter / (time.time() - fps_start_time)
fps_text = f"FPS: {fps:.2f}"

x=0
y=0
formatted_string = f"x: {x}, y: {y}"
print(formatted_string)
while (webcam.isOpened()):
    ret, img = webcam.read()
    if ret == True:
        fps_counter += 1
        
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #dets = detector(rgb_image)
        dets, scores, subdetectors = detector.run(rgb_image, 1, 0)
    
        #for det in dets:
        #    cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255,0,0), 3)
        for i, det in enumerate(dets):
            cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255, 0,0), 2 )
            print("Detection {}, score: {}, face_type: {}".format(det, scores[i], subdetectors[i]))
            x=det.right()-det.left()
            y=det.bottom()-det.top()
            formatted_string = f"x: {x}, y: {y}"

            cv2.putText(img, formatted_string, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if time.time() - fps_start_time >= 1:  # 1초마다 FPS 업데이트
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_text = f"FPS: {fps:.2f}"
                    cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    fps_counter = 0
                    fps_start_time = time.time()
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("WEBCAM", img)

        if cv2.waitKey(1) == 27:
            break

webcam.release()
cv2.destroyAllWindows()
