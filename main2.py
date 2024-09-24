import cv2
import numpy as np
import pickle
import math
from cvzone.ColorModule import ColorFinder
import cvzone

DebugMode = False
colorFinder = ColorFinder(DebugMode)  # Color debug mode

imgListBallsDectected = []
countHit = 0
hitDrawBallInfoList = []
totalScore = 0

RedHsvVals = {'hmin': 0, 'smin': 173, 'vmin': 90, 'hmax': 9, 'smax': 255, 'vmax': 217}
GreenHsvVals = {'hmin': 42, 'smin': 47, 'vmin': 58, 'hmax': 70, 'smax': 255, 'vmax': 255}

# Initialize lists for tracked darts
tracked_red_darts = []
tracked_green_darts = []

class TrackedDart:
    def __init__(self, position, color):
        self.positions = [position]  # 프레임별 다트 위치 리스트
        self.color = color  # 다트 색상 ('red' 또는 'green')
        self.stationary_frames = 0  # 다트가 고정된 상태로 있는 프레임 수
        self.missed_frames = 0  # 다트가 감지되지 않은 프레임 수
        self.is_stationary = False  # 다트가 고정된 상태인지 여부

    def update(self, new_position, movement_threshold, stationary_threshold):
        # 마지막 위치와 새로운 위치 간의 거리 계산
        last_position = self.positions[-1]
        distance = math.hypot(new_position[0] - last_position[0], new_position[1] - last_position[1])

        # 새로운 위치 추가
        self.positions.append(new_position)

        if distance < movement_threshold:
            self.stationary_frames += 1  # 이동이 적을 경우 고정된 상태로 간주
        else:
            self.stationary_frames = 0  # 이동이 있으면 고정 상태 초기화

        # 고정된 다트로 기록할지 여부 결정
        if self.stationary_frames >= stationary_threshold:
            self.is_stationary = True
        else:
            self.is_stationary = False

        self.missed_frames = 0  # 업데이트 되었으므로 미검출 프레임 초기화

def draw_contours_on_image(image, all_contours):
    try:
        output_image = image.copy()
        for contours in all_contours:
            for contour in contours:
                cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
        return output_image
    except Exception as e:
        print(f"Error drawing contours: {e}")
        return image

def find_dartboard_cornerPoints(img): 
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found")
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cornerPoints = [[x, y], [x+w, y], [x, y+h], [x+w, y+h]]
        return cornerPoints
    except Exception as e:
        print(f"Error finding dartboard corner points: {e}")
        return None

def getBoard(img, cornerPoints):
    try:
        if cornerPoints is None or len(cornerPoints) != 4:
            raise ValueError("Invalid corner points")
        width, height = int(400 * 1.5), int(380 * 1.5)
        pts1 = np.float32(cornerPoints)
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (width, height))
        return imgOutput
    except Exception as e:
        print(f"Error getting board image: {e}")
        return img

def detectColorDarts(img, hsvVals):
    imgColor, mask = colorFinder.update(img, hsvVals)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def detect_dart_positions_with_cvzone(img, mask, min_area=250):
    try:
        imgContours, conFound = cvzone.findContours(img, mask, minArea=min_area)
        dart_positions = []

        for contour in conFound:
            center_x, center_y = contour['center']  # 중심 좌표
            dart_positions.append((center_x, center_y))
        return imgContours, dart_positions
    except Exception as e:
        print(f"Error detecting dart positions with cvzone: {e}")
        return img, []

def update_tracked_darts(tracked_darts, detected_positions, color, movement_threshold=5, stationary_threshold=5):
    for position in detected_positions:
        matched = False
        for dart in tracked_darts:
            last_position = dart.positions[-1]
            distance = math.hypot(position[0] - last_position[0], position[1] - last_position[1])
            if distance < movement_threshold:
                dart.update(position, movement_threshold, stationary_threshold)
                matched = True
                break

        if not matched:
            # 새로운 다트를 기록
            new_dart = TrackedDart(position, color)
            tracked_darts.append(new_dart)

    # 다트가 일정 프레임 이상 감지되지 않을 경우 리스트에서 제거
    for dart in tracked_darts:
        if dart.positions[-1] not in detected_positions:
            dart.missed_frames += 1
            if dart.missed_frames > 5:
                tracked_darts.remove(dart)

def filter_similar_positions(stationary_positions, new_position, threshold=10):
    for pos in stationary_positions:
        distance = math.hypot(new_position[0] - pos[0], new_position[1] - pos[1])
        if distance < threshold:
            return False  # 너무 가까운 위치이면 무시
    return True  # 새로운 위치로 간주

# 고정된 다트의 위치를 기록할 리스트
stationary_red_darts = []
stationary_green_darts = []

# 이전에 기록된 고정된 다트의 위치 리스트
previous_stationary_red_darts = []
previous_stationary_green_darts = []


cap = cv2.VideoCapture('Videos/Video3.mp4')
frameCounter = 0

try:
    with open('contours.pkl', 'rb') as f:
        all_contours = pickle.load(f)
except Exception as e:
    print(f"Error loading contours: {e}")
    all_contours = []

cornerPoints = None

while True:
    frameCounter += 1

    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameCounter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()

    if not success:
        print("Failed to read frame from video")
        break

    if cornerPoints is None:
        cornerPoints = find_dartboard_cornerPoints(img)

    # Extract dartboard area from the original image
    imgBoard = getBoard(img, cornerPoints)

    # Detect red and green darts using color detection
    maskRed = detectColorDarts(imgBoard, RedHsvVals)
    maskGreen = detectColorDarts(imgBoard, GreenHsvVals)

    # Detect red dart positions
    redImgContours, redDartPositions = detect_dart_positions_with_cvzone(imgBoard, maskRed, 250)
    # Detect green dart positions
    greenImgContours, greenDartPositions = detect_dart_positions_with_cvzone(imgBoard, maskGreen)

    # Update tracked darts for red and green darts
    update_tracked_darts(tracked_red_darts, redDartPositions, 'red', movement_threshold=10)
    update_tracked_darts(tracked_green_darts, greenDartPositions, 'green')

    # 고정된 다트 위치 추출
    for dart in tracked_red_darts:
        if dart.is_stationary and filter_similar_positions(stationary_red_darts, dart.positions[-1]):
            stationary_red_darts.append(dart.positions[-1])

    for dart in tracked_green_darts:
        if dart.is_stationary and filter_similar_positions(stationary_green_darts, dart.positions[-1]):
            stationary_green_darts.append(dart.positions[-1])

    # 새로운 고정된 다트 위치가 추가된 경우에만 출력
    if stationary_red_darts != previous_stationary_red_darts:
        print("기록된 고정된 빨간 다트 위치:", stationary_red_darts)
        previous_stationary_red_darts = stationary_red_darts.copy()

    if stationary_green_darts != previous_stationary_green_darts:
        print("기록된 고정된 녹색 다트 위치:", stationary_green_darts)
        previous_stationary_green_darts = stationary_green_darts.copy()

    # 고정된 다트 위치 시각화
    for position in stationary_red_darts:
        cv2.circle(imgBoard, (int(position[0]), int(position[1])), 10, (0, 0, 255), cv2.FILLED)
    for position in stationary_green_darts:
        cv2.circle(imgBoard, (int(position[0]), int(position[1])), 10, (0, 255, 0), cv2.FILLED)

    # Display the updated board image
    cv2.imshow("Dartboard with Stationary Darts", imgBoard)

    # Display contour images
    cv2.imshow("Red Dart Contours", redImgContours)
    cv2.imshow("Green Dart Contours", greenImgContours)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()