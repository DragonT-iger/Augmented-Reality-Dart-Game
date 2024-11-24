# 사용하지 않는 기본 템플릿을 위한 버전입니다. 

#main

import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import pickle
import os

print("현재 작업 디렉토리 : ", os.getcwd())

cap = cv2.VideoCapture('Videos/Video3.mp4')
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

with open('contours.pkl', 'rb') as f:
    polygonsWithScore = pickle.load(f)
print(polygonsWithScore)


cornerPoints = [[671, 614], [1585, 581], [675, 1648], [1614, 1626]]

def draw_contours_on_image(image, all_contours):
    # 외곽선을 그릴 이미지를 생성
    output_image = image.copy()

    # 모든 프레임의 외곽선을 그리기
    for contours in all_contours:
        for contour in contours:
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)  # 초록색 외곽선, 두께 2

    return output_image



def getBoard(img):
    width , height = int(400 * 1.5), int(380 * 1.5)
    pts1 = np.float32(cornerPoints)
    pts2 = np.float32([[0 , 0] , [width , 0] , [0, height], [width , height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img , matrix, (width , height))

    for x in range(4):
        cv2.circle(img, (cornerPoints[x][0] , cornerPoints[x][1]) , 15 , (0,255,0),cv2.FILLED)


    # cv2.imwrite('imgOutput.png',imgOutput)

    return imgOutput



def find_dartboard_cornerPoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #그레이스케일

    cv2.imwrite('gray.png',gray)

    # 그레이스케일 이미지를 이진화 이미지로 변환
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    #threst를 저장
    # cv2.imwrite('thresh.png',thresh)

    # 외곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # cv2.imwrite('contours.png',contours)

    # 가장 큰 외곽선 -> 다트보드의 외곽선 
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    cornerPoints = [[x, y], [x+w, y], [x, y+h], [x+w, y+h]]
    
    return cornerPoints

def detectColorDarts(img, hsvVals):
    imgBlur = cv2.GaussianBlur(img, (7, 7), 2)
    imgColor , mask = colorFinder.update(img, hsvVals)

    cv2.imwrite("detectColorDarts1.png", imgColor)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("detectColorDarts2.png", mask)

    return mask




flag = True
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


    if(flag):
        cornerPoints = find_dartboard_cornerPoints(img)
        flag = False


    # imgBoard = getBoard(img)


    redDartMask = detectColorDarts(img, RedHsvVals)
    greenDartMask = detectColorDarts(img, GreenHsvVals)

    
    # cv2.imwrite("redDartMask.png", redDartMask)
    

    

    ### Remove Previous Detections

    for x, img in enumerate(imgListBallsDectected):
        redDartMask = redDartMask - img
        # cv2.imshow(str(x), redDartMask)

    
    cv2.imwrite("redDartMask2.png", redDartMask)


    redImgContours, redConFound = cvzone.findContours(img, redDartMask, 250)
    greenImgContours, blueConFound = cvzone.findContours(img, greenDartMask, 250)

    if redConFound:
        countHit += 1
        if countHit == 3:
            # 원본 마스크는 유지하고, 팽창된 마스크만 따로 생성
            # dilatedMask = cv2.dilate(redDartMask, np.ones((5, 5), np.uint8), iterations=3)

            # 팽창된 마스크를 리스트에 추가
            imgListBallsDectected.append(redDartMask.copy())
            print("Red Dart Detected")
            countHit = 0

            for polyScore in polygonsWithScore:
                center = redConFound[0]['center']

                # print(polyScore[1])                
                polyOutside = np.array([polyScore[0]], np.int32)
                polyInside = np.array([polyScore[1]], np.int32)

                outSide = cv2.pointPolygonTest(polyOutside, center, False)

                print(outSide)

                
                if polyInside is not None and polyInside.size > 2:
                    inSide = cv2.pointPolygonTest(polyInside, center, False)
                else:
                    inSide = -1  # 이는 외부를 의미

                isDartInArea = 0

                if outSide == 1 and inSide == -1:
                    isDartInArea = 1

                print(isDartInArea)
                if isDartInArea == 1:
                    hitDrawBallInfoList.append([redConFound[0]['bbox'], redConFound[0]['center'], polyOutside])
                    totalScore += polyScore[2]

    for bbox, center, poly in hitDrawBallInfoList:
        cv2.rectangle(img, bbox, (0, 255, 0), 2)
        cv2.circle(img, center, 5, (0, 255, 0), cv2.FILLED)
        cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=3)


    print("Red Score: ", totalScore)

    # cv2.imwrite('imgBoard.png',imgBoard)
    # cv2.imshow("Image", img)
    cv2.imshow("Image Board", img)

    # cv2.imshow("Red Mask", redDartMask)
    # cv2.imshow("Green Mask", greenDartMask)
    
    cv2.imshow("Red Contours", redImgContours)
    cv2.imshow("Green Contours", greenImgContours)
    
   
    



    cv2.waitKey(5)
