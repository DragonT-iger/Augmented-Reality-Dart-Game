import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import pickle

def draw_contours_on_image(image, all_contours):
    # 외곽선을 그릴 이미지를 생성
    output_image = image.copy()

    # 모든 프레임의 외곽선을 그리기
    for contours in all_contours:
        for contour in contours:
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)  # 초록색 외곽선, 두께 2

    return output_image

def simplify_contours(contours, max_points=200):
    simplified_contours = []
    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > max_points:
            approx = approx[::len(approx)//max_points][:max_points]
        simplified_contours.append(approx)
    return simplified_contours

def remove_nearby_white_areas(image):
    # 가우시안 블러 적용 (노이즈 제거)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Canny Edge Detection 적용 (엣지 검출)
    edges = cv2.Canny(blurred_image, 50, 150)

    # 외곽선 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 원본 이미지에서 외곽선 근처의 하얀색 영역만 지우기 위한 마스크 생성
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=4)  # 두께를 4로 설정하여 근처 영역 포함

    # 마스크를 적용하여 근처의 하얀색 부분만 지우기
    result_image = cv2.bitwise_and(image, cv2.bitwise_not(mask))

    return result_image, contours

# 비디오 파일 로드
cap = cv2.VideoCapture('Videos/Video3.mp4')
frame_count = 0

# 기존 Pictures/AutoGeneratePolygon 디렉토리 삭제 후 재생성
base_dir = 'Pictures/AutoGeneratePolygon'
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir)

# 외곽선 좌표를 저장할 리스트 초기화
all_contours = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 30 == 0:  # 프레임 간격 조정 (매 30번째 프레임 저장)
        # 프레임별 디렉토리 생성
        frame_dir = os.path.join(base_dir, f'frame_{frame_count}')
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        
        cv2.imwrite(os.path.join(frame_dir, f'original_frame_{frame_count}.jpg'), frame)  # 원본 이미지 저장

        # 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.imwrite(os.path.join(frame_dir, f'gray_frame_{frame_count}.jpg'), gray)  # 그레이스케일 이미지 저장

        # 이진화(흑백 변환) - 임계값 조정
        _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV) # 흑백 이상하면 2번째 파라미터 조정
        cv2.imwrite(os.path.join(frame_dir, f'binary_frame_{frame_count}.jpg'), binary)

        # 외곽선 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 원을 그리기 위한 빈 이미지 생성
        output_image = np.zeros_like(gray)

        # 거의 사각형 모양인 것만 필터링 후 그리기
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # print(f"Contour: x={x}, y={y}, w={w}, h={h}")  # 디버그 정보 출력

            if w > 24 and h > 24:  # 작은 사각형 크기 필터링
                aspect_ratio = float(w) / h
                # print(f"Aspect Ratio: {aspect_ratio}")  # 디버그 정보 출력
                if 0.8 <= aspect_ratio <= 1.2:  # 거의 사각형인 경우만 필터링
                    cv2.drawContours(output_image, [contour], -1, (255, 255, 255), 1)

        # 필터링 결과 저장
        cv2.imwrite(os.path.join(frame_dir, f'filtered_frame_{frame_count}.jpg'), output_image)
        processed_frame, contours = remove_nearby_white_areas(output_image)

        processed_frame, contours = remove_nearby_white_areas(processed_frame)
        

        for i in range(8):
            cv2.imwrite(os.path.join(frame_dir, f'processed_frame_{frame_count}_{i}.jpg'), processed_frame)
            processed_frame, contours = remove_nearby_white_areas(processed_frame)
            simplified_contours = simplify_contours(contours)
            all_contours.append(simplified_contours)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 기본 프레임 이미지에 외곽선 그리기 frame_30
first_frame_image_path = 'Pictures/AutoGeneratePolygon/frame_30/original_frame_30.jpg'
first_frame_image = cv2.imread(first_frame_image_path)

# 모든 외곽선을 그린 이미지 생성
image_with_all_contours = draw_contours_on_image(first_frame_image, all_contours)

for i in range(8):
    # 컨투어를 이미지에 그리기
    image_with_contour = draw_contours_on_image(first_frame_image, [all_contours[i]])
    
    # 결과 이미지 파일 경로 설정
    result_image_path = f'Pictures/AutoGeneratePolygon/frame_30/first_frame_with_contour_{i}.jpg'
    
    # 이미지를 파일로 저장
    cv2.imwrite(result_image_path, image_with_contour)

cap.release()
cv2.destroyAllWindows()

# 외곽선 좌표를 파일로 저장
with open('contours.pkl', 'wb') as f:
    pickle.dump(all_contours, f)
