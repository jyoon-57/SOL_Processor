import cv2
import time  # 시간 체크를 위해 time 모듈 추가
from action_recognizer import ActionRecognizer
import circadian_engine
import context_fusion

def run_sol_system():
    print("SOL Processor를 초기화합니다...")
    recognizer = ActionRecognizer()
    cap = recognizer.start_camera()
    
    # [수정] 함수 형태 모듈 호출이 아닌, 클래스 기반 CircadianEngine 객체를 초기화합니다.
    engine = circadian_engine.CircadianEngine()
    
    previous_action = None
    
    # 0.5초 주기로 콘솔에 출력하기 위해 마지막 출력 시간을 기록할 변수입니다.
    last_print_time = 0.0

    try:
        while True:
            # [Step 1: 시각 인지]
            current_action = recognizer.get_action(cap) # READ, WORK, LIE DOWN, NONE, QUIT
            
            # 종료 신호 감지
            if current_action == "QUIT":
                print("\n>> [종료 신호 감지] 프로그램을 안전하게 종료합니다.")
                break
                
            # [Step 2: 일주기 리듬]
            # [수정] 초기화된 engine 객체의 메서드를 호출하여 조명 딕셔너리 값을 가져옵니다.
            base_light = engine.get_base_lighting()
            
            # [Step 3: 융합 제어]
            # context_fusion이 base_light를 받아 행동(current_action)과 결합합니다.
            final_light = context_fusion.calculate_final_light(current_action, base_light)
            
            # [출력 1] 사용자의 행동(Action)이 변경되었을 때만 특별히 눈에 띄게 알림을 출력합니다.
            if current_action != previous_action:
                print(f"\n>> [행동 인식 변경] 새로운 행동: {current_action}")
                previous_action = current_action
            
            # [출력 2] 0.5초마다 각 조명의 상태와 제어값을 실시간으로 출력합니다.
            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                # 딕셔너리 내부에서 각 구역별 타겟값과 피드백 제어값을 추출합니다.
                main_zone = final_light["zones"]["main"]
                indirect_zone = final_light["zones"]["indirect"]
                task_zone = final_light["zones"]["task"]
                feedback = final_light["feedback"]["wiz_step_percent"]
                
                # 콘솔에 한눈에 보이게끔 한 줄로 포매팅하여 출력합니다.
                print(f"[실시간 제어] Action: {current_action:9s} | "
                      f"Main[Lux:{main_zone['target_lux']:3d}, CCT:{main_zone['target_cct']}K] | "
                      f"Indirect[Lux:{indirect_zone['target_lux']:3d}, CCT:{indirect_zone['target_cct']}K] | "
                      f"Task[Lux:{task_zone['target_lux']:3d}, CCT:{task_zone['target_cct']}K] | "
                      f"제어: {feedback:+d}%")
                
                last_print_time = current_time
            
    except KeyboardInterrupt:
        print("\n>> [강제 종료 감지] 프로그램을 안전하게 종료합니다.")
    finally:
        # 안전한 자원 해제
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("모든 자원이 해제되었습니다.")

if __name__ == "__main__":
    run_sol_system()
