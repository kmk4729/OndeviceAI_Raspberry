import cv2

# JPEG 이미지 파일 경로 설정
image_path = "dataset/User.4.23.jpg"

# JPEG 이미지 읽기
img = cv2.imread(image_path)

# 이미지가 제대로 읽혔는지 확인
if img is None:
    print("이미지를 읽을 수 없습니다.")
else:
    # 이미지의 높이와 너비 가져오기
    height, width, channels = img.shape

    # 이미지의 모든 픽셀 값 출력
    for y in range(height):
        for x in range(width):
            pixel_value = img[y, x]
            print(f"픽셀 위치: ({x}, {y}), 픽셀 값: {pixel_value}")
