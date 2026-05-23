import socket

# 영상을 수신할 포트 번호
UDP_IP = "0.0.0.0"
UDP_PORT = 5555

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"[테스트 시작] {UDP_PORT}번 포트에서 영상 데이터가 들어오는지 감시합니다...")

packet_count = 0
try:
    while True:
        # 데이터가 들어올 때까지 대기
        data, addr = sock.recvfrom(65535) 
        packet_count += 1
        # 100번째 패킷마다 알림을 띄움 (너무 빨리 올라가는 것 방지)
        if packet_count % 100 == 0:
            print(f">>> {addr[0]} 로부터 영상 패킷 수신 중! (누적: {packet_count}개, 크기: {len(data)} bytes)")
except KeyboardInterrupt:
    print("\n[테스트 종료]")