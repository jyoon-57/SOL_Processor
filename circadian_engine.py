import datetime
import math
import socket
import threading
import time
from astral import LocationInfo
from astral.sun import sun
from zoneinfo import ZoneInfo

class CircadianEngine:
    def __init__(self):
        # 서울(Seoul) 위치 정보를 설정합니다. (일출/일몰 계산용)
        self.city = LocationInfo("Seoul", "South Korea", "Asia/Seoul", 37.5665, 126.9780)
        self.timezone = ZoneInfo("Asia/Seoul")
    
        # 날짜별 일출/일몰 시간을 캐싱(저장)하여 불필요한 재계산을 방지합니다.
        self._cached_date = None
        self._start_time = None
        self._dimming_time = None

        # 실시간 센서 조도값을 저장할 변수 (기본값 300.0)
        self.current_lux = 300.0
        
        # UDP 수신 데몬 스레드 시작
        self._listener_thread = threading.Thread(target=self._lux_listener_thread, daemon=True)
        self._listener_thread.start()

    def _lux_listener_thread(self):
        """백그라운드에서 센서(Raspberry Pi)로부터 UDP 조도 데이터를 수신하여 업데이트합니다."""
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", 5000))
                print("[CircadianEngine] UDP 소켓 바인딩 성공 (Port: 5000)")
                
                while True:
                    data, addr = sock.recvfrom(1024)
                    try:
                        message = data.decode('utf-8').strip()
                        if message.startswith("LUX:"):
                            lux_str = message.split(":")[1]
                            self.current_lux = float(lux_str)
                    except (UnicodeDecodeError, ValueError, IndexError):
                        # 파싱 에러 발생 시 무시하고 계속 수신
                        pass
            except OSError as e:
                print(f"[CircadianEngine] 포트 5000 바인딩 실패, 5초 후 재시도... ({e})")
                time.sleep(5)
            except Exception as e:
                print(f"[CircadianEngine] UDP 수신 중 예기치 않은 오류 발생: {e}, 5초 후 재시도...")
                time.sleep(5)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

    def _calculate_anchors(self, current_time: datetime.datetime):
        """실제 일출과 일몰 시간을 기반으로 기상(start)과 소등(dimming) 기준 시간을 계산합니다."""
        current_date = current_time.date()
        
        # 이미 오늘 날짜의 기준 시간을 계산해 두었다면 캐시된 값을 반환합니다.
        if self._cached_date == current_date:
            return self._start_time, self._dimming_time

        # astral 라이브러리를 사용하여 오늘 날짜의 일출/일몰 시간을 구합니다.
        s = sun(self.city.observer, date=current_date, tzinfo=self.timezone)
        
        real_sunrise = s["sunrise"]
        real_sunset = s["sunset"]
        
        # 기상 하한선(06:30)과 소등 하한선(18:00)을 설정합니다.
        limit_start = datetime.datetime.combine(current_date, datetime.time(6, 30), tzinfo=self.timezone)
        limit_dimming = datetime.datetime.combine(current_date, datetime.time(18, 0), tzinfo=self.timezone)
        
        # 여름철 너무 이른 일출, 겨울철 너무 이른 일몰을 방지하기 위해 max 연산을 사용합니다.
        self._start_time = max(real_sunrise, limit_start)
        self._dimming_time = max(real_sunset, limit_dimming)
        self._cached_date = current_date
        
        return self._start_time, self._dimming_time

    def _interpolate_cct(self, cct_start: float, cct_end: float, progress: float) -> int:
        """자연스러운 흑체 궤적(Blackbody curve)을 따라 색온도를 보간합니다. (Mired 단위 사용)"""
        # 진행도(progress)가 0.0 미만이거나 1.0을 초과하지 않도록 제한합니다.
        progress = max(0.0, min(1.0, progress))
        
        # 켈빈(K)을 마이어드(Mired) 단위로 변환합니다. (1,000,000 / K)
        # 마이어드로 변환 후 선형 보간해야 우리 눈에 자연스러운 빛의 변화가 만들어집니다.
        mired_start = 1_000_000.0 / cct_start
        mired_end = 1_000_000.0 / cct_end
        
        mired_current = mired_start + (mired_end - mired_start) * progress
        
        # 다시 켈빈(K) 단위로 되돌립니다.
        cct_current = 1_000_000.0 / mired_current
        return int(round(cct_current))

    def _apply_weather_correction(self, lux: float, cct: int, weather: str):
        """흐림, 비 등 날씨 상태에 따라 조도와 색온도를 보정합니다."""
        if weather in ["Cloudy", "Rain"]:
            lux = lux * 0.8  # 조도는 80% 수준으로 감소
            cct = cct - 200  # 색온도는 200K 감소 (더 따뜻한 빛)
        
        # 너무 낮거나 높은 범위로 벗어나지 않도록 안전 범위를 지정합니다. (WiZ 전구 기준 대략 2700K ~ 6500K)
        cct = max(2700, min(6500, cct))
        lux = max(0.0, lux)
        return lux, cct

    def _calculate_feedback(self, target_lux: float, current_lux: float) -> int:
        """Main 조명의 목표 조도와 센서의 실제 조도를 비교하여 밝기(%) 증감을 결정합니다."""
        deadband = 50  # 오차 허용 범위 (이 범위 내에서는 조명을 조절하지 않음)
        if current_lux < target_lux - deadband:
            return 5   # 실제가 목표보다 어두우면 5% 증가
        elif current_lux > target_lux + deadband:
            return -5  # 실제가 목표보다 밝으면 5% 감소
        return 0

    def get_base_lighting(self, current_time=None, current_lux=None, weather="Clear", current_wiz_percent=50):
        """현재 시간에 맞는 각 조명 구역(Main, Indirect, Task)의 목표 조도와 색온도를 반환합니다."""
        # 외부에서 조도값이 명시되지 않았다면 실시간 수집된 값을 사용합니다.
        if current_lux is None:
            current_lux = self.current_lux

        # 시간 값이 안 들어오면 현재 시간을 기준으로 설정합니다.
        if current_time is None:
            current_time = datetime.datetime.now(self.timezone)
        elif current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=self.timezone)

        # 오늘 날짜의 기준 시간(일출, 일몰)을 가져옵니다.
        start_time, dimming_time = self._calculate_anchors(current_time)
        
        # 시간대별 분기 처리를 위해 기준 시간(오전 10시, 오후 8시 30분 등)을 미리 계산합니다.
        current_date = current_time.date()
        time_10_00 = datetime.datetime.combine(current_date, datetime.time(10, 0), tzinfo=self.timezone)
        time_10_30 = datetime.datetime.combine(current_date, datetime.time(10, 30), tzinfo=self.timezone)
        time_20_30 = datetime.datetime.combine(current_date, datetime.time(20, 30), tzinfo=self.timezone)
        time_22_00 = datetime.datetime.combine(current_date, datetime.time(22, 0), tzinfo=self.timezone)
        
        # 기본값 세팅
        main_lux, indirect_lux, task_lux = 0.0, 0.0, 0.0
        main_cct, indirect_cct, task_cct = 2700, 2700, 2700
        
        # =========================================================
        # 1. 기상 구간 (Sunrise/06:30 ~ 10:00)
        # =========================================================
        if start_time <= current_time < time_10_00:
            duration = (time_10_00 - start_time).total_seconds()
            elapsed = (current_time - start_time).total_seconds()
            progress = elapsed / duration # 진행률 0.0 ~ 1.0
            
            # 간접 조명은 전체 구간에 걸쳐 0에서 500으로 서서히 켜집니다.
            indirect_lux = 500.0 * progress
            
            # 메인 조명은 진행률이 50%를 넘긴 시점부터 켜지기 시작합니다.
            if progress > 0.5:
                main_progress = (progress - 0.5) * 2.0
                main_lux = 500.0 * main_progress
            else:
                main_lux = 0.0
                
            cct = self._interpolate_cct(3000, 6000, progress)
            main_cct = indirect_cct = task_cct = cct

        # =========================================================
        # 2. 주간 구간 (10:00 ~ 일몰/18:00)
        # =========================================================
        elif time_10_00 <= current_time < dimming_time:
            main_lux = indirect_lux = 500.0
            
            # 10:00 ~ 10:30 사이에는 색온도가 6000K에서 5000K로 떨어집니다.
            if current_time < time_10_30:
                duration = (time_10_30 - time_10_00).total_seconds()
                elapsed = (current_time - time_10_00).total_seconds()
                progress = elapsed / duration
                cct = self._interpolate_cct(6000, 5000, progress)
            else:
                cct = 5000
                
            main_cct = indirect_cct = task_cct = cct

        # =========================================================
        # 3. 저녁 구간 (일몰/18:00 ~ 22:00)
        # =========================================================
        elif dimming_time <= current_time < time_22_00:
            duration = (time_22_00 - dimming_time).total_seconds()
            elapsed = (current_time - dimming_time).total_seconds()
            overall_progress = elapsed / duration
            
            # 간접 조명은 500에서 150으로 떨어집니다.
            indirect_lux = 500.0 - (350.0 * overall_progress)
            
            # 메인 조명은 20:30 시점에 정확히 0 lx로 꺼집니다.
            if current_time >= time_20_30:
                main_lux = 0.0
            else:
                main_duration = (time_20_30 - dimming_time).total_seconds()
                main_progress = elapsed / main_duration
                main_lux = 500.0 * (1.0 - main_progress)
                
            cct = self._interpolate_cct(5000, 2700, overall_progress)
            main_cct = indirect_cct = task_cct = cct

        # =========================================================
        # 4. 심야 구간 (22:00 ~ 다음날 기상)
        # =========================================================
        else:
            indirect_lux = 50.0
            main_lux = 0.0
            cct = 2700
            main_cct = indirect_cct = task_cct = cct

        # Task 조명은 circadian_engine에서 무조건 0으로 둡니다.
        # (이후 context_fusion.py에서 별도로 제어됩니다)
        task_lux = 0.0

        # 구역별 최종 조도와 색온도에 날씨 보정 계수를 반영합니다.
        main_lux, main_cct = self._apply_weather_correction(main_lux, main_cct, weather)
        indirect_lux, indirect_cct = self._apply_weather_correction(indirect_lux, indirect_cct, weather)
        task_lux, task_cct = self._apply_weather_correction(task_lux, task_cct, weather)

        # 메인 조명의 타겟 조도(target_lux)와 실제 센서값(current_lux)을 비교하여 피드백을 계산합니다.
        feedback_step = self._calculate_feedback(main_lux, current_lux)

        return {
            "zones": {
                "main": {"target_lux": int(round(main_lux)), "target_cct": main_cct},
                "indirect": {"target_lux": int(round(indirect_lux)), "target_cct": indirect_cct},
                "task": {"target_lux": int(round(task_lux)), "target_cct": task_cct}
            },
            "feedback": {
                "wiz_step_percent": feedback_step
            }
        }

if __name__ == "__main__":
    # =========================================================
    # 테스트 시뮬레이션 코드 (터미널에서 이 파일을 실행하면 작동합니다)
    # =========================================================
    engine = CircadianEngine()
    tz = ZoneInfo("Asia/Seoul")
    today = datetime.datetime.now(tz).date()
    
    test_times = [
        datetime.datetime.combine(today, datetime.time(4, 0), tzinfo=tz),
        datetime.datetime.combine(today, datetime.time(8, 0), tzinfo=tz),
        datetime.datetime.combine(today, datetime.time(10, 15), tzinfo=tz),
        datetime.datetime.combine(today, datetime.time(12, 0), tzinfo=tz),
        datetime.datetime.combine(today, datetime.time(19, 30), tzinfo=tz),
        datetime.datetime.combine(today, datetime.time(21, 0), tzinfo=tz),
        datetime.datetime.combine(today, datetime.time(23, 0), tzinfo=tz)
    ]
    
    print("=== SOL Processor Circadian Engine Simulation ===")
    anchors = engine._calculate_anchors(test_times[0])
    print(f"Sunrise Anchor: {anchors[0].strftime('%H:%M:%S')}")
    print(f"Dimming Anchor: {anchors[1].strftime('%H:%M:%S')}\n")
    
    for t in test_times:
        print(f"--- Time: {t.strftime('%H:%M')} ---")
        res = engine.get_base_lighting(current_time=t, current_lux=300, weather="Clear")
        print(f"Main:     Lux={res['zones']['main']['target_lux']:3d}, CCT={res['zones']['main']['target_cct']}K")
        print(f"Indirect: Lux={res['zones']['indirect']['target_lux']:3d}, CCT={res['zones']['indirect']['target_cct']}K")
        print(f"Task:     Lux={res['zones']['task']['target_lux']:3d}, CCT={res['zones']['task']['target_cct']}K")
        print(f"Feedback (Main target vs 300 current_lux): {res['feedback']['wiz_step_percent']}%")
        print()
