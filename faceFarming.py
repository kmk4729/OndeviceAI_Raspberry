import cv2
import os
import threading
import dlib

# Function to update image count for a given user ID
def update_image_count(txt_file, user_id, count):
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            lines = file.readlines()
        found = False
        with open(txt_file, 'w') as file:
            for line in lines:
                if line.startswith(user_id):
                    file.write(f"{user_id}:{count}\n")
                    found = True
                else:
                    file.write(line)
        if not found:
            with open(txt_file, 'a') as file:
                file.write(f"{user_id}:{count}\n")
    else:
        with open(txt_file, 'w') as file:
            file.write(f"{user_id}:{count}\n")


# Function to capture images for a given user ID
def capture_images(user_id):
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width640
    cam.set(4, 480) # set video height480
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    #face_detector = dlib.get_frontal_face_detector()

    print(f"\n [INFO] Initializing face capture for user ID {user_id}. Look the camera and wait ...")

    # Load or initialize image count from text file
    count_file_path = "count.txt"
    if os.path.exists(count_file_path):
        with open(count_file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith(user_id):
                count = int(line.split(":")[1])
                break
        else:
            count = 0
    else:
        count = 0

    newcount = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            newcount += 1
            # Save the captured image into the datasets folder
            cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif newcount >= 30: # Take 30 face sample and stop video
             break

    # Update or create text file with updated image count
    update_image_count(count_file_path, user_id, count)

    # Do a bit of cleanup
    print(f"\n [INFO] Exiting face capture for user ID {user_id} and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

# Input user IDs for face capture
user_ids = input('\n enter user IDs separated by space: ').split()

# Create and start a thread for each user ID
threads = []
for user_id in user_ids:
    thread = threading.Thread(target=capture_images, args=(user_id,))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("\n [INFO] All face capture processes completed")
