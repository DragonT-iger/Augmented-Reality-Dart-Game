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
        self.positions = [position]  # Positions over frames
        self.color = color  # 'red' or 'green'
        self.stationary_frames = 0  # Frames the dart has been stationary
        self.missed_frames = 0  # Frames the dart was not detected
        self.is_stationary = False

    def update(self, new_position, movement_threshold, stationary_threshold):
        # Calculate movement between last position and new position
        last_position = self.positions[-1]
        distance = math.hypot(new_position[0] - last_position[0],
                              new_position[1] - last_position[1])

        # Update positions
        self.positions.append(new_position)

        if distance < movement_threshold:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0  # Reset if movement is significant

        # Determine if the dart is stationary
        if self.stationary_frames >= stationary_threshold:
            self.is_stationary = True
        else:
            self.is_stationary = False

        self.missed_frames = 0  # Reset missed frames since we've just updated

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
            center_x, center_y = contour['center']  # Center coordinates
            dart_positions.append((center_x, center_y))
        return imgContours, dart_positions
    except Exception as e:
        print(f"Error detecting dart positions with cvzone: {e}")
        return img, []

def update_tracked_darts(tracked_darts, detected_positions, color, movement_threshold=5, stationary_threshold=5):
    for position in detected_positions:
        # Try to match with existing darts
        matched = False
        for dart in tracked_darts:
            last_position = dart.positions[-1]
            distance = math.hypot(position[0] - last_position[0], position[1] - last_position[1])
            if distance < movement_threshold:
                dart.update(position, movement_threshold, stationary_threshold)
                matched = True
                break
        if not matched:
            # Create a new tracked dart
            new_dart = TrackedDart(position, color)
            tracked_darts.append(new_dart)

    # Handle darts that were not detected in this frame
    for dart in tracked_darts:
        if dart.positions[-1] not in detected_positions:
            dart.missed_frames += 1
            if dart.missed_frames > 5:
                # Remove darts that have not been seen for a while
                tracked_darts.remove(dart)

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

    # Draw contours for visualization
    imgWithContours = draw_contours_on_image(img, all_contours)

    # Extract dartboard area with contours for visualization
    imgBoardWithContours = getBoard(imgWithContours, cornerPoints)

    # Display the corner points on the original image for visualization
    imgWithCorners = imgWithContours.copy()
    for point in cornerPoints:
        cv2.circle(imgWithCorners, (point[0], point[1]), 15, (0, 255, 0), cv2.FILLED)

    # Display images
    cv2.imshow("Dartboard with Polygons", imgBoardWithContours)

    if DebugMode:
        cv2.imshow("Original Image", img)
        cv2.imshow("Mask", maskRed)
    else:
        cv2.imshow("Red Darts Mask", maskRed)
        cv2.imshow("Green Darts Mask", maskGreen)

    # Detect red dart positions
    redImgContours, redDartPositions = detect_dart_positions_with_cvzone(imgBoard, maskRed, 125)
    # Detect green dart positions
    greenImgContours, greenDartPositions = detect_dart_positions_with_cvzone(imgBoard, maskGreen)

    # Update tracked darts for red and green darts
    update_tracked_darts(tracked_red_darts, redDartPositions, 'red', movement_threshold=10)
    update_tracked_darts(tracked_green_darts, greenDartPositions, 'green')

    # Collect positions of stationary darts
    stationary_red_darts = [dart.positions[-1] for dart in tracked_red_darts if dart.is_stationary]
    stationary_green_darts = [dart.positions[-1] for dart in tracked_green_darts if dart.is_stationary]

    # Output stationary dart positions
    print("Stationary red dart positions:", stationary_red_darts)
    print("Stationary green dart positions:", stationary_green_darts)

    # Draw stationary darts on the board image
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