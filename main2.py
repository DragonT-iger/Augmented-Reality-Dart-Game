import cv2
import numpy as np
import pickle

def draw_contours_on_image(image, all_contours):
    # 외곽선을 그릴 이미지를 생성
    output_image = image.copy()

    # 모든 프레임의 외곽선을 그리기
    for contours in all_contours:
        for contour in contours:
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)  # 초록색 외곽선, 두께 2

    return output_image

def find_dartboard_cornerPoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이스케일

    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # 외곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 외곽선 -> 다트보드의 외곽선 
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    cornerPoints = [[x, y], [x+w, y], [x, y+h], [x+w, y+h]]
    
    return cornerPoints

def getBoard(img, cornerPoints):
    width, height = int(400 * 1.5), int(380 * 1.5)
    pts1 = np.float32(cornerPoints)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))

    for x in range(4):
        cv2.circle(img, (cornerPoints[x][0], cornerPoints[x][1]), 15, (0, 255, 0), cv2.FILLED)

    return imgOutput

# 비디오 파일 로드
cap = cv2.VideoCapture('Videos/Video3.mp4')
frameCounter = 0

# contours.pkl에서 저장된 외곽선 로드
with open('contours.pkl', 'rb') as f:
    all_contours = pickle.load(f)

cornerPoints = None

while True:
    frameCounter += 1

    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameCounter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()

    if success:
        if cornerPoints is None:
            cornerPoints = find_dartboard_cornerPoints(img)

        # 다트보드 이미지를 보정
        imgBoard = getBoard(img, cornerPoints)

        # 저장된 외곽선을 그려서 이미지 위에 표시
        imgBoardWithContours = draw_contours_on_image(imgBoard, all_contours)

        # 이미지를 imshow로 출력
        cv2.imshow("Dartboard with Polygons", imgBoardWithContours)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
