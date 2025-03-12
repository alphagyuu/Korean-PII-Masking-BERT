#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 전처리 스크립트 (최적화 버전)
-------------------------
이 스크립트는 원본 Korean SNS JSON 파일을 읽어 아래의 전처리 과정을 수행합니다.
  1. 각 대화 항목에서 header의 dialogueID와 body의 발화(utterance)에서 utteranceID, turnID, utterance 추출
  2. 각 발화 텍스트에서 허용된 라벨(ALLOWED_LABELS) 외의 라벨과 그 내용을 제거

프로젝트 폴더 구조 (최상위 폴더: Korean-PII-Masking-BERT):
  data_generation/
      preprocess.py         <-- 이 스크립트 파일
      data/
          Korean_SNS_final_preprocessed/   <-- 전처리 완료 JSON 파일 저장 폴더 (자동 생성)
      Korean_SNS/
          Training/         <-- 원본 JSON 파일들이 위치하는 폴더

실행 예:
    python preprocess.py
"""

import os
import json
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 허용할 라벨 목록 (이름, URL, 소속, 주소, 신원, 계정, 번호, 금융)
ALLOWED_LABELS = {"이름", "URL", "소속", "주소", "신원", "계정", "번호", "금융"}

# 전역 정규표현식 패턴 (한번만 컴파일)
LABEL_PATTERN = re.compile(r'#@([^#]+)#')

def filter_labels(text):
    """
    주어진 텍스트에서 허용되지 않은 라벨과 그 내용을 제거합니다.
    허용된 라벨은 텍스트에 그대로 유지됩니다.
    """
    matches = list(LABEL_PATTERN.finditer(text))
    if not matches:
        return text.strip()

    result_parts = []
    prev_end = 0

    for idx, match in enumerate(matches):
        start, end = match.span()
        label = match.group(1)
        # 매치 전까지의 텍스트 추가
        result_parts.append(text[prev_end:start])
        
        # 다음 매치 시작 위치 결정
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        
        # 허용된 라벨이면 라벨부터 다음 매치 전까지의 텍스트를 포함
        if label in ALLOWED_LABELS:
            result_parts.append(text[start:next_start])
        
        prev_end = next_start

    # 마지막 부분 추가
    result_parts.append(text[prev_end:])
    return ''.join(result_parts).strip()

def process_file(input_file, output_dir):
    """
    하나의 JSON 파일을 처리하는 함수.
    입력 파일을 읽어 전처리 후 지정한 출력 디렉토리에 저장합니다.
    """
    try:
        with input_file.open('r', encoding='utf-8') as f:
            input_data = json.load(f)
    except json.JSONDecodeError as e:
        return f"파일 {input_file.name} 읽기 에러: {e}"

    new_data_list = []

    # 각 대화 데이터 항목 처리
    for item in input_data.get("data", []):
        # header에서 dialogueID 추출
        header = item.get("header", {})
        dialogue_info = header.get("dialogueInfo", {})
        dialogue_id = dialogue_info.get("dialogueID", "")
        new_header = {"dialogueInfo": {"dialogueID": dialogue_id}}

        # body(발화 목록) 처리: 리스트 컴프리헨션 활용
        body = item.get("body", [])
        new_body = [{
            "utteranceID": utterance.get("utteranceID", ""),
            "turnID": utterance.get("turnID", ""),
            "utterance": filter_labels(utterance.get("utterance", ""))
        } for utterance in body]

        new_data_list.append({"header": new_header, "body": new_body})

    final_json_data = {"data": new_data_list}

    # 출력 파일 이름: 원본 파일명에 _final.json 추가 (예: sample_final.json)
    output_filename = input_file.stem + "_final.json"
    output_file_path = output_dir / output_filename

    with output_file_path.open('w', encoding='utf-8') as outfile:
        json.dump(final_json_data, outfile, ensure_ascii=False, indent=2)

    return f"전처리 완료 파일 저장: {output_file_path}"

def main():
    # 현재 작업 디렉토리: data_generation 폴더 내에서 실행한다고 가정
    BASE_DIR = Path.cwd()
    
    # 원본 JSON 파일들이 위치한 디렉토리 (data_generation/Korean_SNS/Training)
    INPUT_DIR = BASE_DIR / "Korean_SNS" / "Training"
    
    # 최종 전처리 완료된 파일을 저장할 디렉토리 (data_generation/data/Korean_SNS_final_preprocessed)
    OUTPUT_DIR = BASE_DIR / "data" / "Korean_SNS_final_preprocessed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_DIR.exists():
        print(f"입력 디렉토리 {INPUT_DIR} 가 존재하지 않습니다.")
        return

    json_files = list(INPUT_DIR.glob('*.json'))
    if not json_files:
        print("처리할 JSON 파일이 없습니다.")
        return

    # 파일 단위 병렬 처리 (ProcessPoolExecutor 사용)
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, input_file, OUTPUT_DIR): input_file for input_file in json_files}
        for future in as_completed(future_to_file):
            result = future.result()
            print(result)

if __name__ == "__main__":
    main()
