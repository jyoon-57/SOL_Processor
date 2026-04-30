import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

def verify_code_integrity(target_file, keyword):
    """
    파일의 물리적 상태와 내용을 검증하여 AI 할루시네이션을 판별합니다.
    """
    # 1. 파일 경로 설정 (이 스크립트가 있는 현재 폴더 기준)
    base_dir = Path(__file__).resolve().parent
    full_path = base_dir / target_file
    
    print(f"\n🔍 [SOL 무결성 검사] Target: {target_file}")
    
    # 2. 파일 존재 여부 확인
    if not full_path.exists():
        print(f"❌ [에러] 파일이 존재하지 않습니다: {full_path.name}")
        sys.exit(1)

    # 3. 최근 수정 시간 확인 (AI가 실제로 파일을 건드렸는지 물리적 증거 확보)
    file_mtime = datetime.fromtimestamp(full_path.stat().st_mtime)
    now = datetime.now()
    time_diff = now - file_mtime

    print(f"   - 마지막 수정 시간: {file_mtime.strftime('%H:%M:%S')}")
    print(f"   - 현재 시간: {now.strftime('%H:%M:%S')}")
    
    # 60초 이내에 수정되지 않았다면 작성하지 않은 것으로 간주
    if time_diff > timedelta(seconds=60):
        print(f"⚠️  [할루시네이션 경고] 파일이 최근 60초 이내에 수정되지 않았습니다!")
        print(f"   (마지막 수정 후 {time_diff.seconds}초 경과. AI가 말로만 수정했다고 거짓말했을 확률이 높습니다.)")
        sys.exit(2)

    # 4. 핵심 키워드 존재 여부 확인
    content = full_path.read_text(encoding='utf-8')
    if keyword in content:
        print(f"✅ [검증 완료] '{keyword}' 코드가 파일에 정상적으로 반영되었습니다.")
        sys.exit(0)
    else:
        print(f"❌ [실패] 파일은 방금 수정되었으나, 요청한 핵심 코드 '{keyword}'가 누락되었습니다.")
        sys.exit(3)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("💡 사용법: python sol_verifier.py [파일명] [확인할키워드]")
        print("예시: python sol_verifier.py sol_processor.py deque")
        sys.exit(1)
    
    verify_code_integrity(sys.argv[1], sys.argv[2])