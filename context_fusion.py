# context_fusion.py
# (신규) 두뇌의 역할. '행동'과 '기본 조명값'을 융합해 '최종 명령' 생성.

def calculate_final_light(current_action, base_light):
    # TODO: 행동과 기본 조명을 융합하는 실제 로직 구현 필요
    # 임시 로직: 행동이 READ이면 집중 모드로 임시 변경
    cct = base_light.get("cct", 3000)
    bri = base_light.get("bri", 50)
    
    if current_action == "READ":
        cct = 4000
        bri = 80
    elif current_action == "WORK":
        cct = 5000
        bri = 100
        
    return {"cct": cct, "bri": bri}
