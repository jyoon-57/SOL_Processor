import cv2
from action_recognizer import ActionRecognizer
import circadian_engine
import context_fusion

def run_sol_system():
    print("SOL Processor를 초기화합니다...")
    recognizer = ActionRecognizer()
    cap = recognizer.start_camera()
    
    try:
        while True:
            # [Step 1: 시각 인지]
            current_action = recognizer.get_action(cap) #READ, WORK, LIE DOWN, NONE, QUIT
            
            # 종료 신호 감지
            if current_action == "QUIT":
                print("\n>> [종료 신호 감지] 프로그램을 안전하게 종료합니다.")
                break
                
            # [Step 2: 일주기 리듬]
            base_light = circadian_engine.get_base_lighting()
            
            # [Step 3: 융합 제어]
            final_light = context_fusion.calculate_final_light(current_action, base_light)
            
            # 제어 결과 출력
            cct = final_light.get("cct")
            bri = final_light.get("bri")
            print(f"[SOL Control] Action: {current_action} | Final Light: CCT {cct}K, BRI {bri}%")
            
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
