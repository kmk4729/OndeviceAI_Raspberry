import cv2
import os
import threading
import dlib

# Function to update image count and user name for a given user ID
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

# Function to save landmarks to a file
# Function to save landmarks to a file



# Function to capture images for a given user ID
def capture_images(user_id):
    cam = cv2.VideoCapture(0)
    cam.set(3, 320) # set video width
    cam.set(4, 240) # set video height

    # Initialize HOG face detector from dlib
    face_detector = dlib.get_frontal_face_detector()
    # Initialize landmark predictor
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces using HOG face detector
        faces = face_detector(gray)
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
                face_img_resized = cv2.resize(face_img_normalized, (60, 60))
                # Multiply by 255 to get back to original scale
                face_img_resized *= 255
                # Save the image with proper filename
                image_filename = f"dataset/User_{user_id}/User_{user_id}_{count}.jpg"
                os.makedirs(os.path.dirname(image_filename), exist_ok=True)
                cv2.imwrite(image_filename, face_img_resized)

                # Detect landmarks
                landmarks = landmark_predictor(gray, face)
                # Save landmarks to file
                save_landmarks(user_id, count, landmarks)

            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif newcount >= 30: # Take 30 face samples and stop video
             break

    # Update or create text file with updated image count and user name
    update_image_count(count_file_path, user_id, user_name, count)

    # Do a bit of cleanup
    print(f"\n [INFO] Exiting face capture for user ID {user_id} and cleaning up")
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
