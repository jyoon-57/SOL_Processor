# context_fusion.py
# (신규) 두뇌의 역할. '행동'과 '기본 조명값'을 융합해 '최종 명령' 생성.
import copy

def calculate_final_light(current_action, base_light):
    """
    circadian_engine에서 도출된 일주기 리듬 기본 조명값(base_light)과
    action_recognizer에서 추론한 현재 사용자 행동(current_action)을 융합합니다.
    """
    # 원본 딕셔너리가 훼손되지 않도록 깊은 복사(deepcopy)를 수행합니다.
    final_light = copy.deepcopy(base_light)
    
    # circadian_engine에서 넘겨준 시간대 정보 (없으면 DAY로 간주)
    phase = base_light.get("phase", "DAY")
    
    # 행동 문자열 정규화 (None -> NONE)
    action = "NONE" if current_action == "None" else current_action

    # =========================================================================
    # A. 메인 조명 (Main Lighting)
    # =========================================================================
    if phase == "NIGHT":
        final_light["zones"]["main"]["target_lux"] = 0
        final_light["feedback"]["wiz_step_percent"] = 0
    else: # MORNING, DAY, EVENING
        if action == "NONE":
            pass # 원본 값 및 feedback 유지
        elif action in ["READ", "WORK"]:
            # 밝기 등은 유지하되 깜빡임(Hunting) 방지를 위해 피드백 잠금
            final_light["feedback"]["wiz_step_percent"] = 0
        elif action in ["LIE DOWN", "SLEEP", "NO_HUMAN"]:
            # 시야 간섭 원천 차단 및 에너지 절약
            final_light["zones"]["main"]["target_lux"] = 0
            final_light["feedback"]["wiz_step_percent"] = 0

    # =========================================================================
    # B. 간접 조명 (Indirect Lighting)
    # =========================================================================
    indirect = final_light["zones"]["indirect"]
    
    if action == "NO_HUMAN":
        if phase == "NIGHT":
            indirect["target_lux"] = 0
        else:
            indirect["target_lux"] = int(round(indirect["target_lux"] * 0.2)) # 대기 모드
    elif action == "SLEEP":
        if phase in ["NIGHT", "EVENING", "DAY"]:
            indirect["target_lux"] = 0 # 완전 소등 (빛 공해 차단 및 낮잠 배려)
        elif phase == "MORNING":
            pass # 자연스러운 기상을 위해 원본 값(0->500 서서히 밝아짐) 유지
    elif action == "LIE DOWN":
        if phase == "EVENING":
            indirect["target_lux"] = 300
            indirect["target_cct"] = 2700
        elif phase == "NIGHT":
            indirect["target_lux"] = 50
            indirect["target_cct"] = 2700
        else: # MORNING, DAY
            indirect["target_lux"] = int(round(indirect["target_lux"] * 0.5))
    elif action == "WORK":
        if phase == "NIGHT":
            indirect["target_lux"] = 50
            indirect["target_cct"] = 4000 # 작업 시 색온도 이질감 방지
        else:
            pass # 원본 수용
    elif action in ["NONE", "READ"]:
        if phase == "NIGHT":
            indirect["target_lux"] = 50
            indirect["target_cct"] = 2700
        else:
            pass # 원본 수용

    # =========================================================================
    # C. 국부 조명 (Task Lighting)
    # =========================================================================
    task = final_light["zones"]["task"]
    
    if action in ["NONE", "LIE DOWN", "SLEEP", "NO_HUMAN"]:
        task["target_lux"] = 0
    elif action == "READ":
        if phase in ["MORNING", "DAY"]:
            task["target_lux"] = 500
            task["target_cct"] = 4000
        elif phase == "EVENING":
            task["target_lux"] = 300
            task["target_cct"] = 3000
        elif phase == "NIGHT":
            task["target_lux"] = 150
            task["target_cct"] = 2700
    elif action == "WORK":
        if phase in ["MORNING", "DAY"]:
            task["target_lux"] = 500
            task["target_cct"] = 5000
        elif phase in ["EVENING", "NIGHT"]:
            task["target_lux"] = 400
            task["target_cct"] = 4000

    return final_light

if __name__ == "__main__":
    # 간단한 시뮬레이션용 코드
    import pprint
    
    # 임의의 base_light 템플릿
    def get_mock_base_light(phase):
        return {
            "zones": {
                "main": {"target_lux": 500, "target_cct": 5000},
                "indirect": {"target_lux": 500, "target_cct": 5000},
                "task": {"target_lux": 0, "target_cct": 2700}
            },
            "feedback": {
                "wiz_step_percent": 5
            },
            "phase": phase
        }
        
    test_cases = [
        ("MORNING", "SLEEP"),
        ("DAY", "SLEEP"),
        ("DAY", "WORK"),
        ("EVENING", "LIE DOWN"),
        ("NIGHT", "READ"),
        ("DAY", "NO_HUMAN")
    ]
    
    for phase, action in test_cases:
        mock_base = get_mock_base_light(phase)
        res = calculate_final_light(action, mock_base)
        print(f"--- [ Phase: {phase:7} | Action: {action:9} ] ---")
        print(f"Main    : Lux={res['zones']['main']['target_lux']:3d}, CCT={res['zones']['main']['target_cct']}, Feed={res['feedback']['wiz_step_percent']}")
        print(f"Indirect: Lux={res['zones']['indirect']['target_lux']:3d}, CCT={res['zones']['indirect']['target_cct']}")
        print(f"Task    : Lux={res['zones']['task']['target_lux']:3d}, CCT={res['zones']['task']['target_cct']}")
        print()
