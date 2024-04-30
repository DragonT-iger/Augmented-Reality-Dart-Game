#main

import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import pickle
import os

print("현재 작업 디렉토리 : ", os.getcwd())

cap = cv2.VideoCapture('Videos/Video1.mp4')
frameCounter = 0

colorFinder = ColorFinder(False)

# CornerPoint는 임시로 설정한 값
# 671 614  # 1585 581
# 675 1648 # 1614 1626

# 빨간색 다트 검출 {'hmin': 0, 'smin': 194, 'vmin': 90, 'hmax': 7, 'smax': 255, 'vmax': 217}
# 초록색 다트 검출 {'hmin': 42, 'smin': 47, 'vmin': 58, 'hmax': 74, 'smax': 255, 'vmax': 255}

RedHsvVals = {'hmin': 0, 'smin': 194, 'vmin': 90, 'hmax': 7, 'smax': 255, 'vmax': 217}
GreenHsvVals = {'hmin': 42, 'smin': 47, 'vmin': 58, 'hmax': 74, 'smax': 255, 'vmax': 255}


imgListBallsDectected = []
countHit = 0
hitDrawBallInfoList = []
totalScore = 0

with open('polygon', 'rb') as f:
    polygonsWithScore = pickle.load(f)
print(polygonsWithScore)


cornerPoints = [[671, 614], [1585, 581], [675, 1648], [1614, 1626]]

def getBoard(img):
    width , height = int(400 * 1.5), int(380 * 1.5)
    pts1 = np.float32(cornerPoints)
    pts2 = np.float32([[0 , 0] , [width , 0] , [0, height], [width , height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img , matrix, (width , height))

    for x in range(4):
        cv2.circle(img, (cornerPoints[x][0] , cornerPoints[x][1]) , 15 , (0,255,0),cv2.FILLED)

    return imgOutput



def find_dartboard_cornerPoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #그레이스케일

    cv2.imwrite('gray.png',gray)

    # 그레이스케일 이미지를 이진화 이미지로 변환
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    #threst를 저장
    cv2.imwrite('thresh.png',thresh)

    # 외곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imwrite('contours.png',contours)

    # 가장 큰 외곽선 -> 다트보드의 외곽선 
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    cornerPoints = [[x, y], [x+w, y], [x, y+h], [x+w, y+h]]
    
    return cornerPoints

def detectColorDarts(img, hsvVals):
    imgBlur = cv2.GaussianBlur(img, (7, 7), 2)
    imgColor , mask = colorFinder.update(img, hsvVals)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


while True:

    frameCounter += 1

    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameCounter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img  = cap.read()

    # image_path = "imgBoard.png"
    # img = cv2.imread(image_path)

    if img is None:
        print("Image not found")
        break



    cornerPoints = find_dartboard_cornerPoints(img)
    
    imgBoard = getBoard(img)


    redDartMask = detectColorDarts(imgBoard, RedHsvVals)
    greenDartMask = detectColorDarts(imgBoard, GreenHsvVals)

    

    redImgContours, redConFound = cvzone.findContours(imgBoard, redDartMask, 250)
    greenImgContours, blueConFound = cvzone.findContours(imgBoard, greenDartMask, 250)







    ### Remove Previous Detections

    for x, img in enumerate(imgListBallsDectected):
        redDartMask = redDartMask - img
        # cv2.imshow(str(x), redDartMask)


    

    if redConFound:
        countHit += 1
    if countHit == 10:
            imgListBallsDectected.append(redDartMask)
            print("Red Dart Detected")
            countHit = 0

            for polyScore in polygonsWithScore:
                center = redConFound[0]['center']
                poly = np.array([polyScore[0]], np.int32)
                inside = cv2.pointPolygonTest(poly, center, False)
                # print(inside)
                if inside:
                    hitDrawBallInfoList.append([redConFound[0]['bbox'], redConFound[0]['center'], poly])
                    totalScore += polyScore[2]

    for bbox, center, poly in hitDrawBallInfoList:
        cv2.rectangle(imgBoard, bbox, (0, 255, 0), 2)
        cv2.circle(imgBoard, center, 5, (0, 255, 0), cv2.FILLED)
        cv2.polylines(imgBoard, [poly], isClosed=True, color=(0, 255, 0), thickness=5)


    print("Total Score: ", totalScore)

    # cv2.imwrite('imgBoard.png',imgBoard)
    # cv2.imshow("Image", img)
    cv2.imshow("Image Board", imgBoard)

    # cv2.imshow("Red Mask", redDartMask)
    # cv2.imshow("Green Mask", greenDartMask)
    
    cv2.imshow("Red Contours", redImgContours)
    cv2.imshow("Green Contours", greenImgContours)
    



    cv2.waitKey(5)




