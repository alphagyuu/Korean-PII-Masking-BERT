#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare Batch Input Script for GPT API Inference
--------------------------------------------------
이 스크립트는 전처리된 Korean SNS JSON 파일(예: Korean_SNS_pp1_8labels)을 읽어  
• 로컬 가명정보(이름, URL, 주소, 번호)는 즉시 생성하고,  
• GPT API 추론이 필요한 태그(금융, 소속, 계정, 신원)에 대해 발화 및 인접 발화(컨텍스트)를 추출하여  
  LLM 입력 데이터를 생성한 후, 배치 API 요청(배치 입력)을 JSONL 파일로 저장합니다.
  
프로젝트 구조 (최상위 폴더: Korean-PII-Masking-BERT):
  inference/
  models/
      Korean-PII-Masking-CRF/
      Korean-PII-Masking-BertForTokenClassification/
  train/
      preprocess.py         <-- 이전 전처리 스크립트
      prepare_batch_input.py  <-- 이 스크립트
      data/
          fake_pii/         <-- 모든 CSV 파일들이 저장됨
      Korean_SNS/
          Training/        <-- 원본 JSON 파일들
      Korean_SNS_pp1_8labels/  <-- 전처리 완료 JSON 파일들 (이전 단계 산출물)
  3_generation/
      Korean_SNS_generated1_Training/  <-- 임시 산출물(가명정보 생성 전 단계)
      llm_input/
          extracted_dialogues/         <-- LLM 입력 데이터 (JSONL)
          batchapi/                    <-- GPT API 배치 입력 파일들이 저장됨
"""

import os
import json
import re
import glob
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures  # 병렬 처리를 위해 추가

###########################
# 디렉토리 및 경로 설정
###########################
BASE_DIR = Path.cwd()
INPUT_JSON_DIR = BASE_DIR / "data" / "Korean_SNS_final_preprocessed"
OUTPUT_JSON_DIR = BASE_DIR / "data" / "Korean_SNS_intermediate_generated"
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
LLM_INPUT_DIR = BASE_DIR / "llm_input"
EXTRACTED_DIR = LLM_INPUT_DIR / "extracted_dialogues"
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
BATCH_INPUT_DIR = LLM_INPUT_DIR / "batchapi"
BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "data" / "fake_pii"

###########################
# 가명정보 생성용 로컬 데이터 로드
###########################
balanced_names_path = DATA_DIR / "koreannames_top_1500.csv"
hangul_surname_path = DATA_DIR / "hangul_surname_top_10.csv"
balanced_names_df = pd.read_csv(balanced_names_path)
names = balanced_names_df['Name'].tolist()
hangul_surname_df = pd.read_csv(hangul_surname_path, encoding='utf-8')
surnames = hangul_surname_df.iloc[:, 0].tolist()

company_df = pd.read_csv(DATA_DIR / "filtered_korean_companies.csv", header=None)
company_names = company_df.iloc[:, 0].tolist()
school_df = pd.read_csv(DATA_DIR / "filtered_korean_schools.csv", header=None)
school_names = school_df.iloc[:, 0].tolist()
major_df = pd.read_csv(DATA_DIR / "filtered_korean_major.csv", header=None)
majors = major_df.iloc[:, 0].tolist()
department_df = pd.read_csv(DATA_DIR / "filtered_departments.csv", header=None)
departments = department_df.iloc[:, 0].tolist()
domain_df = pd.read_csv(DATA_DIR / "email.csv", header=None)
domains = domain_df.iloc[:, 0].tolist()

###########################
# 가명정보 생성용 클래스 및 함수 정의
###########################
class KoreanNameGenerator:
    def __init__(self, surnames, names):
        self.surnames = surnames
        self.names = names
        self.total_names = len(names)
        self.used_name_indices = set()
        self.current_index = 0
        self.names_ending_with_consonant = [name for name in names if self.ends_with_consonant(name)]
        self.names_ending_with_vowel = [name for name in names if not self.ends_with_consonant(name)]

    def has_final_consonant(self, char):
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            return ((code - 0xAC00) % 28) != 0
        return False

    def ends_with_consonant(self, name):
        return self.has_final_consonant(name[-1])
    
    def generate_korean_name(self, with_sur=True):
        if len(self.used_name_indices) == self.total_names:
            self.used_name_indices.clear()
            self.current_index = 0
        start_index = self.current_index
        while self.current_index in self.used_name_indices:
            self.current_index = (self.current_index * 31 + 7) % self.total_names
            if self.current_index == start_index:
                self.used_name_indices.clear()
                self.current_index = 0
                break
        self.used_name_indices.add(self.current_index)
        name = self.names[self.current_index]
        surname = self.surnames[self.current_index % len(self.surnames)]
        self.current_index = (self.current_index * 31 + 7) % self.total_names
        return f"{surname}{name}" if with_sur else name

class CompanyNameGenerator:
    def __init__(self, company_names):
        self.company_names = company_names
        self.available = list(company_names)
        random.shuffle(self.available)
    def get_next_company(self):
        if not self.available:
            self.available = list(self.company_names)
            random.shuffle(self.available)
        return self.available.pop()

class SchoolNameGenerator:
    def __init__(self, school_names):
        self.school_names = school_names
        self.available = list(school_names)
        random.shuffle(self.available)
    def get_next_school(self):
        if not self.available:
            self.available = list(self.school_names)
            random.shuffle(self.available)
        return self.available.pop()

class MajorGenerator:
    def __init__(self, majors):
        self.majors = majors
        self.available = list(majors)
        random.shuffle(self.available)
    def generate_major(self):
        if not self.available:
            self.available = list(self.majors)
            random.shuffle(self.available)
        return self.available.pop()

class DepartmentGenerator:
    def __init__(self, departments):
        self.departments = departments
        self.available = list(departments)
        random.shuffle(self.available)
    def generate_department(self):
        if not self.available:
            self.available = list(self.departments)
            random.shuffle(self.available)
        return self.available.pop()

def generate_random_url():
    protocols = ['http', 'https']
    subdomains = ['', 'www', 'blog', 'shop']
    tlds = ['com', 'net', 'org', 'co.kr', 'kr']
    path = '/'.join(''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=random.randint(3, 10)))
                    for _ in range(random.randint(0, 3)))
    domain = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10)))
    url = f"{random.choice(protocols)}://"
    if random.choice([True, False]):
        sub = random.choice(subdomains)
        if sub:
            url += f"{sub}."
    url += f"{domain}.{random.choice(tlds)}"
    if path:
        url += f"/{path}"
    return url

def generate_random_number_string(include_dash=False, length=3):
    if length < 3 or length > 8:
        raise ValueError("length must be between 3 and 8")
    digits = ''.join(str(random.randint(0, 9)) for _ in range(length))
    if include_dash and length >= 5:
        pos = random.randint(1, length - 2)
        return digits[:pos] + '-' + digits[pos:]
    return digits

def generate_bank():
    banks = ['KB국민', '국민', '우리', 'SC제일', '제일', '한국씨티', '씨티',
             'iM뱅크', '하나', '신한', '케이뱅크', '카카오뱅크', '토스뱅크',
             '수협', 'NH농협', '농협', '부산', '경남', '광주', '전북', '제주']
    return random.choice(banks)

def generate_account_number():
    fmt = random.choice([
        "YYY-ZZZZZZZZ-XXX",
        "XXX-YY-ZZZZZZC",
        "XXX-BBBBBB-YY-ZZC",
        "XXXXYY-ZZ-ZZZZZC",
        "XXX-ZZZZZZ-ZZCYY"
    ])
    return ''.join(random.choice('0123456789') if c.isalpha() else c for c in fmt)

def generate_id():
    length = random.randint(7, 12)
    return ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length))

def generate_email(domains):
    length = random.randint(7, 12)
    username = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length))
    domain = random.choice(domains)
    return f"{username}@{domain}"

def generate_security_code():
    return ''.join(random.choices("0123456789", k=6))

# 인스턴스 생성 (글로벌 상태)
name_generator = KoreanNameGenerator(surnames, names)
company_generator = CompanyNameGenerator(company_names)
school_generator = SchoolNameGenerator(school_names)
major_generator = MajorGenerator(majors)
department_generator = DepartmentGenerator(departments)

###########################
# 텍스트 내 라벨 처리 함수
###########################
LABEL_PATTERN = re.compile(r'#@([^#]+)#')
ALLOWED_LOCAL_TAGS = {"이름", "URL", "주소", "번호"}

def filter_and_generate(text):
    tag_values = []
    new_text_parts = []
    pos = 0
    for match in LABEL_PATTERN.finditer(text):
        start, end = match.span()
        label = match.group(1)
        new_text_parts.append(text[pos:start])
        if label in ALLOWED_LOCAL_TAGS:
            if label == "이름":
                fake = name_generator.generate_korean_name()
            elif label == "URL":
                fake = generate_random_url()
            elif label == "주소":
                fake = "가짜주소"  # 실제 주소 생성 로직으로 대체 가능
            elif label == "번호":
                fake = generate_random_number_string(include_dash=True, length=random.randint(3, 8))
            else:
                fake = ""
            tag_values.append(fake)
            new_text_parts.append(match.group(0))
        else:
            tag_values.append("")
            new_text_parts.append(match.group(0))
        pos = end
    new_text_parts.append(text[pos:])
    return ''.join(new_text_parts).strip(), tag_values

###########################
# LLM 입력용 컨텍스트 생성 함수
###########################
def build_context(dialogue_body, target_idx, min_word_count=30):
    context = []
    word_count = 0
    idx = target_idx
    while idx >= 0 and word_count < min_word_count:
        utt = dialogue_body[idx]
        utt_text = utt.get("utterance", "")
        context.insert(0, {
            "utteranceID": utt.get("utteranceID", ""),
            "turnID": utt.get("turnID", ""),
            "utterance": utt_text
        })
        word_count += len(utt_text.split())
        idx -= 1
    idx = target_idx + 1
    while idx < len(dialogue_body) and word_count < min_word_count:
        utt = dialogue_body[idx]
        utt_text = utt.get("utterance", "")
        context.append({
            "utteranceID": utt.get("utteranceID", ""),
            "turnID": utt.get("turnID", ""),
            "utterance": utt_text
        })
        word_count += len(utt_text.split())
        idx += 1
    return context

###########################
# 배치 입력 생성을 위한 LLM 요청 생성 함수
###########################
TAG_OPTIONS = {
    '금융': ['계좌번호', '은행명', '보안코드'],
    '소속': ['회사', '학교', '학과', '업무부서'],
    '계정': ['아이디', '이메일', '웹주소']
}
MODEL = "gpt-4o-mini"

def generate_category_request(llm_item, idx_counter):
    tag = llm_item['tag']
    options = TAG_OPTIONS.get(tag, [])
    system_message = (
        f"너는 주어진 대화에서 마스킹된 단어의 원래 단어 범주를 추론하는 AI assistant야.\n"
        f"주어진 대화에서 #@{tag}#은 원래 단어를 가린 마스크입니다. "
        f"선택지는 다음과 같습니다: {options}. 정답이 없으면 '기타'로 응답하세요."
    )
    dialogue_lines = [f"{utt['turnID']}: {utt['utterance']}" for utt in llm_item['context']]
    dialogue_text = "\n".join(dialogue_lines)
    request = {
        "custom_id": f"request-{idx_counter}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": dialogue_text.strip()}
            ],
            "max_tokens": 35,
            "temperature": 0.5
        }
    }
    return request

def generate_etcs_request(llm_item, idx_counter):
    tag = llm_item['tag']
    instruction = "구체적인 정보"
    system_message = (
        f"너는 주어진 대화에서 마스킹된 위치의 가명정보 문자열을 생성하는 AI assistant야.\n"
        f"마스킹된 원래 단어는 {instruction}입니다. 전체 대화 맥락을 고려하여 자연스러운 가명정보를 단일 단어로 생성하세요."
    )
    dialogue_lines = [f"{utt['turnID']}: {utt['utterance']}" for utt in llm_item['context']]
    dialogue_text = "\n".join(dialogue_lines)
    request = {
        "custom_id": f"request-{idx_counter}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": dialogue_text.strip()}
            ],
            "max_tokens": 15,
            "temperature": 0.8,
            "frequency_penalty": 1.2,
            "presence_penalty": 0.7
        }
    }
    return request

###########################
# 파일별 처리 함수 (병렬 처리 대상)
###########################
def process_file(input_file):
    local_llm_input_data = []
    new_dialogues = []
    try:
        with input_file.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {input_file.name}: {e}")
        return new_dialogues, local_llm_input_data

    dialogues = data.get("data", [])
    for dialogue in dialogues:
        new_dialogue = {"header": dialogue.get("header", {}), "body": []}
        body = dialogue.get("body", [])
        for idx, utt in enumerate(body):
            orig_text = utt.get("utterance", "")
            new_text, tag_values = filter_and_generate(orig_text)
            utt["utterance"] = new_text
            tags = LABEL_PATTERN.findall(orig_text)
            for tag_idx, tag in enumerate(tags):
                if tag not in ALLOWED_LOCAL_TAGS:
                    context = build_context(body, idx)
                    local_llm_input_data.append({
                        "dialogueID": dialogue.get("header", {}).get("dialogueInfo", {}).get("dialogueID", ""),
                        "utteranceID": utt.get("utteranceID", ""),
                        "tag": tag,
                        "index_in_utterance": tag_idx,
                        "context": context
                    })
            new_dialogue["body"].append(utt)
        new_dialogues.append(new_dialogue)

    output_filename = input_file.stem + "_generation_temp1.json"
    output_filepath = OUTPUT_JSON_DIR / output_filename
    with output_filepath.open('w', encoding='utf-8') as fout:
        json.dump({"data": new_dialogues}, fout, ensure_ascii=False, indent=2)
    return new_dialogues, local_llm_input_data

###########################
# 메인 처리: 병렬 JSON 파일 처리 및 LLM 입력 데이터 추출
###########################
all_llm_input_data = []
json_files = list(INPUT_JSON_DIR.glob('*.json'))

with concurrent.futures.ThreadPoolExecutor() as executor:
    # tqdm을 위해 futures.as_completed 사용
    futures = {executor.submit(process_file, file): file for file in json_files}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing preprocessed JSON files"):
        _, file_llm_input = future.result()
        all_llm_input_data.extend(file_llm_input)

llm_input_file = EXTRACTED_DIR / "llm_input_extracted.jsonl"
with llm_input_file.open('w', encoding='utf-8') as fout:
    for item in all_llm_input_data:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"LLM 입력 데이터 저장 완료: {llm_input_file}")

###########################
# 배치 API 요청 생성 (카테고리 및 기타)
###########################
batch_requests_category = []
batch_requests_etcs = []
req_counter = 1
for item in all_llm_input_data:
    if item['tag'] in TAG_OPTIONS:
        batch_requests_category.append(generate_category_request(item, req_counter))
    else:
        batch_requests_etcs.append(generate_etcs_request(item, req_counter))
    req_counter += 1

batch_category_file = BATCH_INPUT_DIR / "batch_input_category.jsonl"
with batch_category_file.open('w', encoding='utf-8') as fout:
    for req in batch_requests_category:
        fout.write(json.dumps(req, ensure_ascii=False) + "\n")
print(f"카테고리 배치 입력 파일 저장: {batch_category_file}")

batch_etcs_file = BATCH_INPUT_DIR / "batch_input_etcs.jsonl"
with batch_etcs_file.open('w', encoding='utf-8') as fout:
    for req in batch_requests_etcs:
        fout.write(json.dumps(req, ensure_ascii=False) + "\n")
print(f"기타(가명정보) 배치 입력 파일 저장: {batch_etcs_file}")

print("=== GPT API 배치 입력 준비 완료 ===")
