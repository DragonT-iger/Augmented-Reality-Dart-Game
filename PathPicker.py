import cv2
import numpy as np
import pickle

path_outer = []  # 외부 다각형 경로
path_inner = []  # 내부 다각형 경로
polygon = []  # 저장할 폴리곤 정보
current_path = path_outer  # 현재 수정 중인 경로

# polygon 파일에서 다각형 정보 읽기
# try:
#     with open('polygon', 'rb') as f:
#         polygon = pickle.load(f)
# except FileNotFoundError:
#     pass

cap = cv2.VideoCapture('Videos/Video1.mp4')
frameCounter = 0


cornerPoints = [[671, 614], [1585, 581], [675, 1648], [1614, 1626]]


def find_dartboard_cornerPoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #그레이스케일


    # 그레이스케일 이미지를 이진화 이미지로 변환
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # 외곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 외곽선 -> 다트보드의 외곽선 
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    cornerPoints = [[x, y], [x+w, y], [x, y+h], [x+w, y+h]]
    
    return cornerPoints


def getBoard(img):
    width , height = int(400 * 1.5), int(380 * 1.5)
    pts1 = np.float32(cornerPoints)
    pts2 = np.float32([[0 , 0] , [width , 0] , [0, height], [width , height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img , matrix, (width , height))

    for x in range(4):
        cv2.circle(img, (cornerPoints[x][0] , cornerPoints[x][1]) , 15 , (0,255,0),cv2.FILLED)

    return imgOutput


def mousePoints(event, x, y, flags, params):
    global current_path
    if event == cv2.EVENT_LBUTTONDOWN:
        current_path.append([x, y])

def draw_polygon(img, path_outer, path_inner):
    if path_outer:
        pts_outer = np.array(path_outer, np.int32)
        pts_outer = pts_outer.reshape((-1, 1, 2))
        cv2.polylines(img, [pts_outer], isClosed=True, color=(255, 0, 0), thickness=5)
    if path_inner:
        pts_inner = np.array(path_inner, np.int32)
        pts_inner = pts_inner.reshape((-1, 1, 2))
        cv2.polylines(img, [pts_inner], isClosed=True, color=(0, 255, 0), thickness=5)

while True:
    success, img = cap.read()

    frameCounter += 1

    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameCounter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cornerPoints = find_dartboard_cornerPoints(img)

    img = getBoard(img)
    if not success:
        break

    draw_polygon(img, path_outer, path_inner)

    cv2.imshow("Video", img)
    cv2.setMouseCallback("Video", mousePoints)
    key = cv2.waitKey(1)

    if key == ord('q'):  # 프로그램 종료
        break
    elif key == ord('c'):  # 내부 다각형 설정으로 변경
        current_path = path_inner
    elif key == ord('o'):  # 외부 다각형 설정으로 변경
        current_path = path_outer
    elif key == ord('s'):  # 다각형 저장
        score = int(input("Enter the score: "))
        polygon.append([path_outer.copy(), path_inner.copy(), score])
        print("Total polygons: ", len(polygon))
        path_inner.clear()
        path_inner = path_outer.copy() # 외부 다각형을 내부 다각형으로 복사
        path_outer.clear()
        current_path = path_outer  # 외부 다각형부터 다시 시작

cv2.destroyAllWindows()
cap.release()

# 저장한 다각형을 파일에 저장
with open('polygon', 'wb') as f:
    pickle.dump(polygon, f)
