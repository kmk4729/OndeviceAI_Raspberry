import cv2
import numpy as np
import os
# 빈 320x240 이미지 생성 (검은 배경)
image = np.zeros((240, 320), dtype=np.uint8)
file_path1=f"land0.txt"

# 주어진 좌표
land1=[]
if os.path.exists(file_path1):
        with open(file_path1, 'r') as file1:
            lines1 = file1.readlines()
        for line1 in lines1:
            land1.append((int(line1.split(" ")[0]),int(line1.split(" ")[1])))

# 하얀 점을 찍음
for coord in land1:
    cv2.circle(image, coord, 1, (255, 255, 255), -1)

# 이미지 저장
cv2.imwrite('output_image.png', image)

# 결과 이미지 출력 (optional)
cv2.imshow('Image with White Dots', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
