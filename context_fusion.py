# context_fusion.py
# (신규) 두뇌의 역할. '행동'과 '기본 조명값'을 융합해 '최종 명령' 생성.
import copy

def calculate_final_light(current_action, base_light):
    """
    circadian_engine에서 도출된 일주기 리듬 기본 조명값(base_light)과
    action_recognizer에서 추론한 현재 사용자 행동(current_action)을 융합합니다.
    """
    # 원본 딕셔너리(base_light)가 훼손되지 않도록 깊은 복사(deepcopy)를 수행합니다.
    final_light = copy.deepcopy(base_light)
    
    # [융합 로직]
    # 사용자의 행동이 READ(독서)이거나 WORK(작업)인 경우, 
    # 원래 0 lx로 꺼져 있던 Task(국부 조명) 구역을 켜주어 집중할 수 있도록 돕습니다.
    if current_action == "READ":
        final_light["zones"]["task"]["target_lux"] = 500
        final_light["zones"]["task"]["target_cct"] = 4000  # 독서하기 편안한 따뜻한 백색
    elif current_action == "WORK":
        final_light["zones"]["task"]["target_lux"] = 500
        final_light["zones"]["task"]["target_cct"] = 5000  # 작업에 집중하기 좋은 주백색
        
    # 만약 "NONE"이나 "LIE DOWN" 등 집중이 필요 없는 행동이라면 
    # circadian_engine이 내려준 Task 조명 타겟값(0 lx)을 그대로 유지합니다.
    
    # Main, Indirect 구역과 feedback 값 또한 일주기 리듬(base_light)의 결정사항을 그대로 따릅니다.
    return final_light
