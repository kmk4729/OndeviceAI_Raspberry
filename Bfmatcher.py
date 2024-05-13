import cv2
import os
import numpy as np
# 이미지 불러오기
count=0
while(count<10):
    img1 = cv2.imread(f'dataset/test1/test{count}.jpg')
    img2 = cv2.imread(f'dataset/test2/test{count}.jpg')
    file_path1=f"dataset/test1/land{count}.txt"
    file_path2=f"dataset/test2/land{count}.txt"

    # SIFT 특징점 추출기 생성

    # 각 이미지에서 특징점 검출 및 디스크립터 계산

    # Brute-Force Matcher 객체 생성

    # 특징점 매

    # 좋은 매칭 필터링


    # 매칭된 특징점 선으로 연결

    # 이미지를 가로로 붙이기
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    result = cv2.hconcat([img1, img2])
    land1=[]
    land2=[]
    if os.path.exists(file_path1):
            with open(file_path1, 'r') as file1:
                lines1 = file1.readlines()
            for line1 in lines1:
                land1.append((int(line1.split(" ")[0]),int(line1.split(" ")[1])))
    if os.path.exists(file_path2):
            with open(file_path2, 'r') as file2:
                lines2 = file2.readlines()
            for line2 in lines2:
                land2.append((int(line2.split(" ")[0]),int(line2.split(" ")[1])))
    """       
    print(land1)
    print("")
    print(land2)
    """
    for i in range(68):
        
        # 두 점 사이에 선 그리기
        pt1 = (int(land1[i][0]), int(land1[i][1]))
        pt2 = (int(land2[i][0]) + 320, int(land2[i][1]))
        cv2.line(result, pt1, pt2, (0, 255, 0), 1)
    numland1 = np.array(land1, dtype=np.float32)
    numland2 = np.array(land2, dtype=np.float32)

    retval, mask = cv2.findHomography(numland1, numland2, cv2.RANSAC)
    h, w = 240,320

    H_inv =  np.linalg.inv(retval)
    img4 = cv2.warpPerspective(img2, H_inv, (w, h))

    # 결과 출력
    #cv2.imshow('Matches', result)
    result1 = cv2.hconcat([img1,img4])
    result2 = cv2.hconcat([ img2,img4])
    #cv2.imshow('res1', result1)
    #cv2.imshow('res', img2)
    resultresult = cv2.vconcat([result1,result2])

    cv2.imshow('res3', resultresult)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    count+=1
