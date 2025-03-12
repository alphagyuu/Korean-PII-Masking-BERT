#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process GPT API Responses and Merge Final Pseudonym Data
---------------------------------------------------------
이 스크립트는 GPT API 호출 후 생성된 배치 응답 파일들을 읽어  
• 카테고리 추론 및 기타 가명정보 생성 결과를 병합하고,  
• 전처리 후 임시 산출물(데이터/ Korean_SNS_intermediate_generated)과 결합하여 최종 완성 JSON 파일(데이터/ Korean_SNS_final_generated)을 생성합니다.
또한, 첫 번째 스크립트에서 생성된 GPT API 전용 배치 입력 파일들을 삭제합니다.

프로젝트 폴더 구조 (최상위 폴더: Korean-PII-Masking-BERT):
  data_generation/
      process_gpt_api_responses.py  <-- 이 스크립트 파일
      llm_input/
          extracted_dialogues/         <-- LLM 입력 JSONL (사용 후 삭제 가능)
          batchapi/                    <-- 배치 입력 파일 (최종 처리 후 삭제)
      llm_output/
          batchapi/                    <-- GPT API 응답 배치 파일들이 저장됨 (JSONL)
      data/
          Korean_SNS_intermediate_generated/  <-- 전처리 후 임시 산출물 (prepare_batch_input.py 산출)
          Korean_SNS_final_generated/         <-- 최종 결과 JSON 파일이 저장됨
          fake_pii/                          <-- 모든 CSV 파일들이 저장됨
"""

import os
import json
import pandas as pd
from pathlib import Path

###########################
# 디렉토리 및 경로 설정
###########################
BASE_DIR = Path.cwd()

# GPT API 응답 배치 파일들이 저장된 디렉토리 (data_generation/llm_output/batchapi)
BATCH_OUTPUT_DIR = BASE_DIR / "llm_output" / "batchapi"
# LLM 입력 배치 파일들이 저장된 디렉토리 (삭제 대상)
BATCH_INPUT_DIR = BASE_DIR / "llm_input" / "batchapi"

# 전처리 후 임시 산출물 (가명정보 생성 전 단계) -> data/Korean_SNS_intermediate_generated
TEMP_GENERATED_DIR = BASE_DIR / "data" / "Korean_SNS_intermediate_generated"
# 최종 완성 파일 저장 디렉토리 -> data/Korean_SNS_final_generated
FINAL_OUTPUT_DIR = BASE_DIR / "data" / "Korean_SNS_final_generated"
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CSV 데이터는 이제 data/fake_pii 에 저장됨 (필요한 경우 사용)
CSV_DIR = BASE_DIR / "data" / "fake_pii"

###########################
# 배치 응답 병합 함수 (카테고리)
###########################
def merge_category_responses(batch_output_files):
    combined_results = {}
    for file in batch_output_files:
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                data = json.loads(line)
                custom_id = data.get("custom_id", "")
                try:
                    assistant_reply = data["response"]["body"]["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError):
                    assistant_reply = ""
                combined_results[custom_id] = assistant_reply
    return combined_results

###########################
# 배치 응답 병합 함수 (기타)
###########################
def merge_etcs_responses(batch_output_files):
    combined_results = {}
    for file in batch_output_files:
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                data = json.loads(line)
                custom_id = data.get("custom_id", "")
                try:
                    assistant_reply = data["response"]["body"]["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError):
                    assistant_reply = ""
                combined_results[custom_id] = assistant_reply
    return combined_results

###########################
# 최종 결과 JSON 생성 함수
###########################
def merge_final_results(category_results, etcs_results):
    final_dialogues = []
    for file in TEMP_GENERATED_DIR.glob('*_generation_temp1.json'):
        with file.open('r', encoding='utf-8') as fin:
            data = json.load(fin)
        for dialogue in data.get("data", []):
            for utt in dialogue.get("body", []):
                if "originalWords" in utt:
                    updated_words = []
                    for word in utt["originalWords"]:
                        if word == "":
                            custom_id = f"request-{utt.get('utteranceID', '0')}"
                            fake_info = category_results.get(custom_id, etcs_results.get(custom_id, ""))
                            updated_words.append(fake_info)
                        else:
                            updated_words.append(word)
                    utt["originalWords"] = updated_words
            final_dialogues.append(dialogue)
    final_data = {"data": final_dialogues}
    final_file = FINAL_OUTPUT_DIR / "Korean_SNS_final_generated.json"
    with final_file.open('w', encoding='utf-8') as fout:
        json.dump(final_data, fout, ensure_ascii=False, indent=2)
    print(f"최종 결과 파일 저장 완료: {final_file}")

###########################
# 메인 처리
###########################
if __name__ == "__main__":
    batch_cat_files = list(BATCH_OUTPUT_DIR.glob("batch_output_category_*.jsonl"))
    batch_etcs_files = list(BATCH_OUTPUT_DIR.glob("batch_output_etcs_*.jsonl"))
    
    print("배치 응답 병합 중 (카테고리)...")
    category_results = merge_category_responses(batch_cat_files)
    print("배치 응답 병합 중 (기타)...")
    etcs_results = merge_etcs_responses(batch_etcs_files)

    merge_final_results(category_results, etcs_results)

    # 사용한 배치 입력 파일 삭제
    for file in BATCH_INPUT_DIR.glob("*.jsonl"):
        try:
            file.unlink()
            print(f"삭제됨: {file}")
        except Exception as e:
            print(f"삭제 실패 {file}: {e}")

    print("모든 GPT API 응답 병합 및 후처리 완료.")
