import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import math
from collections import deque

# ==============================================================================
# [설정 및 초기화]
# ==============================================================================

# 1. MediaPipe Pose (최신 Tasks API) 초기화
# 다운로드 받은 pose_landmarker_lite.task 모델을 로드합니다.
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

# 뼈대 렌더링 도구 (수동 구현: 최신 API에서는 수동으로 그려주어야 함)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def draw_landmarks_on_image(rgb_image, detection_result):
    if not detection_result.pose_landmarks:
        return rgb_image
    
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape
    
    # 첫 번째 사람의 랜드마크만 그림
    landmarks = detection_result.pose_landmarks[0]
    
    # 뼈대 선 그리기
    for connection in POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_lm = landmarks[start_idx]
        end_lm = landmarks[end_idx]
        
        # 가시성이 충분할 때만 선을 그림
        if getattr(start_lm, 'visibility', 1.0) > 0.5 and getattr(end_lm, 'visibility', 1.0) > 0.5:
            start_pt = (int(start_lm.x * w), int(start_lm.y * h))
            end_pt = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(annotated_image, start_pt, end_pt, (0, 255, 0), 2)
            
    # 관절 점 그리기
    for lm in landmarks:
        if getattr(lm, 'visibility', 1.0) > 0.5:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(annotated_image, pt, 4, (0, 0, 255), -1)
            
    return annotated_image

# 2. 행동을 인식하는 깊은(Deep) AI 모델과 행동 이름이 적힌 파일 경로
MODEL_PATH = "resnet-34_kinetics.onnx"
LABEL_PATH = "action_recognition_kinetics.txt"

# 3. 행동 이름(라벨)들을 파일에서 읽어서 리스트로 저장
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 4. 행동 인식 모델(OpenCV DNN)을 메모리에 불러오기
print("행동 인식 모델을 불러오고 있습니다. 잠시만 기다려주세요...")
net = cv2.dnn.readNet(MODEL_PATH)

# ==============================================================================
# [핵심 파라미터 및 가중치 설정]
# ==============================================================================

# 행동을 분석하기 위해 모아둘 프레임(사진)의 개수 및 프레임 샘플링 설정
FRAME_BUFFER_SIZE = 16
FRAME_SKIP = 2 # 2~3프레임당 1장씩 버퍼에 담아 1~1.5초간의 맥락을 확보 (Frame Skip)
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
frame_counter = 0 # 프레임 샘플링용 카운터

# 시스템 상태 변수
STATE_STABLE = "STABLE"       # 안정 상태: 큰 움직임이 없어 기존 행동을 유지하는 상태
STATE_ANALYZING = "ANALYZING" # 분석 상태: 큰 움직임이 감지되어 깊은 AI가 분석을 시작한 상태
current_state = STATE_STABLE  # 초기 상태는 '안정 상태'
last_known_action = "None"    # 마지막으로 확인된 구체적인 행동

# 움직임을 판단하는 기준점 (역치)
MOVEMENT_THRESHOLD = 0.05 
CONFIDENCE_THRESHOLD = 0.6 # AI의 추론 정확도가 60% 미만이면 무시 (찍기 방지)
previous_landmarks = None # 바로 이전 장면의 관절 위치를 기억하는 변수

# 디자이너님의 요청에 따른 관절별 민감도(가중치) 설정
WEIGHTS = {
    0: 2.0,   # 코 (머리의 움직임)
    11: 1.5,  # 왼쪽 어깨
    12: 1.5,  # 오른쪽 어깨
    23: 1.5,  # 왼쪽 골반
    24: 1.5,  # 오른쪽 골반
    13: 0.2,  # 왼쪽 팔꿈치
    14: 0.2,  # 오른쪽 팔꿈치
    15: 0.1,  # 왼쪽 손목
    16: 0.1,  # 오른쪽 손목
    27: 0.1,  # 왼쪽 발목
    28: 0.1   # 오른쪽 발목
}

# ==============================================================================
# [핵심 알고리즘 함수]
# ==============================================================================

def calculate_movement_delta(prev_marks, curr_marks):
    """이전 장면과 현재 장면의 관절 위치를 비교하여 수치(Delta)로 계산합니다."""
    if prev_marks is None or curr_marks is None:
        return 0.0
    
    total_delta = 0.0
    
    for idx, weight in WEIGHTS.items():
        prev_point = prev_marks[idx]
        curr_point = curr_marks[idx]
        
        if getattr(prev_point, 'visibility', 1.0) < 0.5 or getattr(curr_point, 'visibility', 1.0) < 0.5:
            continue
            
        dist = math.sqrt((curr_point.x - prev_point.x)**2 + (curr_point.y - prev_point.y)**2)
        total_delta += dist * weight
        
    return total_delta

# ==============================================================================
# [메인 프로그램 실행]
# ==============================================================================

cap = cv2.VideoCapture(1)
prev_time = 0
print("카메라 연동 완료. 움직임 분석을 시작합니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 화면을 가져올 수 없습니다.")
        break
        
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # MediaPipe 처리를 위해 BGR을 RGB로 변환하고 mp.Image 객체 생성
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # 뼈대 위치 찾아내기
    detection_result = detector.detect(mp_image)
    
    delta = 0.0
    
    # 1. 원본 프레임 따로 저장 (DNN 입력용, 뼈대 그림 없음)
    clean_frame = frame.copy()
    
    # 뼈대를 성공적으로 찾았다면
    if detection_result.pose_landmarks:
        # 화면에 뼈대 그리기 (사용자 모니터링 화면용)
        frame = draw_landmarks_on_image(frame, detection_result)
        
        # 첫 번째 사람의 랜드마크를 가져옴
        curr_landmarks = detection_result.pose_landmarks[0]
        
        # 이전 화면과 비교하여 움직임 변화량(Delta)을 계산
        delta = calculate_movement_delta(previous_landmarks, curr_landmarks)
        previous_landmarks = curr_landmarks
        
    # [상태 전이 로직]
    # 버퍼링 도중 취소 방지: 현재 STABLE 상태일 때만 delta를 확인하여 ANALYZING으로 진입.
    # 이미 ANALYZING 상태라면 16장이 다 모일 때까지 STABLE로 돌아가지 않음.
    if current_state == STATE_STABLE and delta > MOVEMENT_THRESHOLD:
        current_state = STATE_ANALYZING
        frame_counter = 0 # 프레임 샘플링 카운터 초기화
        
    # [이벤트 기반 심층 분석] 상태가 'ANALYZING'일 때 프레임 수집
    if current_state == STATE_ANALYZING:
        frame_counter += 1
        
        # 2. 프레임 샘플링(Frame Skip): FRAME_SKIP 간격마다 1장씩 버퍼에 담음
        if frame_counter % FRAME_SKIP == 0:
            frame_buffer.append(clean_frame) # 무조건 깨끗한 원본만 버퍼에 담음
        
        # 약속된 개수(16장)가 다 모이면 추론 시작
        if len(frame_buffer) == FRAME_BUFFER_SIZE:
            # OpenCV DNN 전처리
            blob = cv2.dnn.blobFromImages(frame_buffer, 1.0, (112, 112), (114.7748, 107.7354, 99.475), swapRB=True, crop=True)
            blob = np.transpose(blob, (1, 0, 2, 3))
            blob = np.expand_dims(blob, axis=0)
            
            # 모델 추론
            net.setInput(blob)
            outputs = net.forward()
            
            class_id = np.argmax(outputs)
            confidence = outputs[0][class_id]
            
            # 확신도(Confidence) 커트라인 검사 (찍기 방지)
            if confidence >= CONFIDENCE_THRESHOLD:
                last_known_action = labels[class_id]
                print(f">> [새로운 맥락 발견] '{last_known_action}' 행동 감지! (정확도: {confidence:.2f})")
            else:
                print(f"-- [판단 보류] 정확도 미달 ({confidence:.2f} < {CONFIDENCE_THRESHOLD}). 기존 상태 유지.")
            
            # 추론 완료 후: 3. 버퍼를 비우고 상태를 다시 STABLE로 초기화 (중도 취소 방지 로직과 세트)
            frame_buffer.clear()
            current_state = STATE_STABLE

    # ==============================================================================
    # [화면 표시 정보] (모니터링용)
    # ==============================================================================
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Delta: {delta:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    state_color = (0, 0, 255) if current_state == STATE_ANALYZING else (255, 0, 0)
    cv2.putText(frame, f"State: {current_state}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    cv2.putText(frame, f"Action: {last_known_action}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

    cv2.imshow("SOL Processor (Hybrid Prototyping Phase 1)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
