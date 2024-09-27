import cv2

# RTSP 스트림 URL (핸드폰의 RTSP 주소)
rtsp_url = 'rtsp://<핸드폰_IP>:<포트번호>/'

# 스트림 열기
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("RTSP 스트림을 열 수 없습니다.")
    exit()

# 실시간 영상 처리 루프
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("영상을 받을 수 없습니다.")
        break
    
    # 프레임을 처리하는 코드 (원하는 처리를 여기에 추가할 수 있습니다)
    # 예시: 영상을 흑백으로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 실시간 영상 출력
    cv2.imshow('RTSP Live Stream', gray_frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
