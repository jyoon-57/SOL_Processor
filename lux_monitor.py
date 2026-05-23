#라즈베리파이에서 조도 데이터 잘 받고 있나 테스트

import socket

# 노트북이 조도 데이터를 받을 포트 (sol_lens.py의 LUX_SEND_PORT와 일치해야 함)
LISTEN_PORT = 5000

def start_monitoring():
    # UDP 소켓 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 모든 IP(0.0.0.0)에서 들어오는 LISTEN_PORT(5000) 데이터를 수신하도록 바인딩
    sock.bind(("0.0.0.0", LISTEN_PORT))

    print(f"[모니터링 시작] {LISTEN_PORT}번 포트에서 조도 데이터를 기다립니다...\n")

    try:
        while True:
            # 데이터 수신 대기 (최대 1024바이트)
            data, addr = sock.recvfrom(1024)
            # 수신된 데이터를 문자열로 디코딩
            message = data.decode('utf-8')

            # 화면에 출력
            print(f"[{addr[0]}] 에서 수신됨: {message}")

    except KeyboardInterrupt:
        print("\n[모니터링 종료]")
    finally:
        sock.close()

if __name__ == "__main__":
    start_monitoring()