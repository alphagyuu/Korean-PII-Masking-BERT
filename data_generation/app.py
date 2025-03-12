#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_generation/app.py
----------------------
이 스크립트는 data_generation 폴더 내의 다음 스크립트들을 순차적으로 실행합니다.
  1. preprocess.py
  2. prepare_batch_input.py
  3. process_gpt_api_responses.py

각 스크립트는 해당 폴더 내에서 독립적으로 실행 가능하도록 작성되어 있으며,
여기서 subprocess를 사용하여 순차 실행합니다.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    script_path = Path(__file__).parent / script_name
    print(f"실행 중: {script_name}")
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"에러 발생 ({script_name}):")
        print(result.stderr)
        sys.exit(result.returncode)
    else:
        print(result.stdout)

def main():
    run_script("preprocess.py")
    run_script("prepare_batch_input.py")
    run_script("process_gpt_api_responses.py")
    print("data_generation 단계 모든 스크립트 실행 완료.")

if __name__ == "__main__":
    main()
