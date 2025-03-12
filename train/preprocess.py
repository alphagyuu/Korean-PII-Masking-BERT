#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 전처리 및 학습 데이터 생성 스크립트
---------------------------------------
이 스크립트는 두 단계로 진행됩니다.

[1] Standard Form 생성 (Part A)
    - data_generation/data/Korean_SNS_final_generated 폴더 내의 최종 생성 JSON 파일을 읽어
      각 대화를 처리한 후, 라벨 치환이 완료된 “standard form” JSON 파일을 생성합니다.
    - 생성된 파일은 현재 train 폴더 내의 data/standard_form 폴더에 저장됩니다.
    
[2] BERT 학습 데이터 생성 (Part B)
    - 위에서 생성된 standard form JSON 파일들 중 하나의 숫자 폴더(예: “11081”)를 선택하여
      해당 폴더 내 파일들을 토큰화(슬라이딩 윈도우 기법 적용)하고, BIO 라벨을 부여한 뒤
      학습에 사용할 TSV 파일과 원본 문장 및 라벨 정보를 pickle 파일로 생성합니다.
      
프로젝트 폴더 구조 (최상위 폴더: Korean-PII-Masking-BERT):
  ├── data_generation/
  │      └── data/
  │            ├── fake_pii/               # CSV 파일들 (원본은 공개하지 않음)
  │            └── Korean_SNS_final_generated/  <-- 최종 생성 JSON 파일 (이전에 생성됨)
  ├── train/
  │      ├── preprocess.py                <-- 이 스크립트 (학습 데이터 생성)
  │      └── data/
  │            └── standard_form/         <-- Standard Form JSON 및 학습 데이터가 저장됨
  └── (기타 폴더들)

실행 예:
    python preprocess.py
"""

import os
import json
import re
import csv
import pickle
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizerFast

###########################################
# Part A: Standard Form 생성 함수들
###########################################

def load_json(file_path):
    """JSON 파일을 로드하는 함수"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """JSON 파일을 저장하는 함수"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_dialogues(original_data):
    """원본 데이터를 처리하여 최종 standard form JSON을 생성하는 함수"""
    # 각 대화별로 originalWords를 추출하기 위한 딕셔너리 생성
    labels_dict = {}
    for dialogue in original_data.get('data', []):
        dialogue_id = dialogue['header']['dialogueInfo']['dialogueID']
        labels_dict[dialogue_id] = {}
        for utterance in dialogue.get('body', []):
            utterance_id = utterance['utteranceID']
            labels_dict[dialogue_id][utterance_id] = utterance.get('originalWords', [])
    
    final_data = {'data': []}
    for dialogue in original_data.get('data', []):
        dialogue_id = dialogue['header']['dialogueInfo']['dialogueID']
        final_dialogue = {
            'header': dialogue['header'],
            'body': []
        }
        for utterance in dialogue.get('body', []):
            utterance_id = utterance['utteranceID']
            turn_id = utterance['turnID']
            original_text = utterance['utterance']
            original_words = labels_dict.get(dialogue_id, {}).get(utterance_id, [])
            word_index = 0
            cleaned_text = ""
            labels = []
            label_positions = []
            current_pos = 0
            pattern = re.compile(r'#@(\w+)#')
            last_end = 0
            for match in pattern.finditer(original_text):
                start, end = match.span()
                label_type = match.group(1)
                pre_text = original_text[last_end:start]
                cleaned_text += pre_text
                current_pos += len(pre_text)
                if word_index < len(original_words):
                    word = original_words[word_index]
                    word_index += 1
                    cleaned_text += word
                    labels.append(label_type)
                    label_positions.append(f"{current_pos}-{current_pos+len(word)}")
                    current_pos += len(word)
                last_end = end
            cleaned_text += original_text[last_end:]
            final_utterance = {
                "utteranceID": utterance_id,
                "turnID": turn_id,
                "utterance": cleaned_text
            }
            if labels:
                final_utterance["labels"] = labels
                final_utterance["labelPositions"] = label_positions
            final_dialogue['body'].append(final_utterance)
        final_data['data'].append(final_dialogue)
    return final_data

def get_unique_index(directory, base_name):
    """
    고유한 인덱스를 생성하여 파일 이름과 라벨을 반환하는 함수.
    파일명은 {base_name}_standard_form_{prefix}{unique_index}.json 형태로 생성됩니다.
    """
    current_time = datetime.now().strftime("%m%d")
    prefix = f"{current_time}"
    if directory.exists():
        existing_files = [p.name for p in directory.iterdir()]
    else:
        existing_files = []
    unique_index = 1
    while True:
        unique_filename = f"{base_name}_standard_form_{prefix}{unique_index}.json"
        if unique_filename not in existing_files:
            label = f"{prefix}{unique_index}"
            return {"unique_filename": unique_filename, "label": label}
        unique_index += 1

def generate_standard_form():
    """
    Part A:
    - 원본 최종 생성 JSON 파일들을 읽어, 각 대화를 처리한 후 standard form JSON 파일을 생성합니다.
    - 원본 파일들은 data_generation/data/Korean_SNS_final_generated 폴더에서 로드하며,
      결과는 현재 스크립트가 위치한 train/data/standard_form 폴더에 저장됩니다.
    """
    project_root = Path.cwd().parent  # Korean-PII-Masking-BERT
    # 원본 파일 경로 (data_generation/data/Korean_SNS_final_generated)
    original_dir = project_root / 'data_generation' / 'data' / 'Korean_SNS_final_generated'
    if not original_dir.exists():
        print("원본 파일 폴더가 존재하지 않습니다:", original_dir)
        return
    # 출력 폴더: train/data/standard_form
    output_dir = Path.cwd() / 'data' / 'standard_form'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일명 접미사 (원본 파일들은 예를 들어 *_final_generated.json 형태로 저장되어 있다고 가정)
    original_suffix = '_final_generated.json'
    processed_count = 0
    skipped_files = []
    
    for original_file in tqdm(original_dir.glob(f'*{original_suffix}'), desc="Standard Form 생성", unit="file"):
        base_name = original_file.stem.replace(original_suffix[:-5], "")  # 예: 원본이 sample_final_generated.json -> base_name: sample
        try:
            original_json = load_json(original_file)
            final_json = process_dialogues(original_json)
            # 고유한 파일명 생성 (출력 폴더 내의 하위 디렉토리로 저장)
            file_info = get_unique_index(output_dir, base_name)
            output_filename = file_info["unique_filename"]
            subfolder = output_dir / file_info["label"]
            subfolder.mkdir(parents=True, exist_ok=True)
            output_path = subfolder / output_filename
            save_json(final_json, output_path)
            processed_count += 1
            print(f"✅ 처리 완료: {output_filename} (저장 경로: {output_path})")
        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({base_name}): {e}")
            skipped_files.append(base_name)
    print(f"\n전체 처리된 파일 수: {processed_count}")
    if skipped_files:
        print(f"처리되지 않은 파일 ({len(skipped_files)}): {skipped_files}")
    return output_dir  # Standard Form 파일들이 저장된 상위 폴더

###########################################
# Part B: BERT 학습 데이터 생성 함수
###########################################

def sliding_window_split(tokens, labels, max_length=300, stride=50):
    """
    슬라이딩 윈도우 방식으로 토큰과 라벨을 분할합니다.
    """
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i+max_length]
        chunk_labels = labels[i:i+max_length]
        chunks.append((chunk_tokens, chunk_labels))
        if i+max_length >= len(tokens):
            break
    return chunks

def generate_bert_input():
    """
    Part B:
    - 생성된 standard form JSON 파일들 중, 숫자로 구성된 하위 폴더 중 하나(예: "11081")를 선택하여
      해당 폴더 내의 JSON 파일들을 BERT 입력 데이터(TSV 파일 및 pickle 파일)로 변환합니다.
    - 출력은 현재 train 폴더 내의 'Input_for_kcBERT_{selected_folder_name}' 폴더에 저장됩니다.
    """
    # 입력 폴더: train/data/standard_form
    input_dir = Path.cwd() / 'data' / 'standard_form'
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더가 존재하지 않습니다: {input_dir}")
    subdirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not subdirs:
        raise FileNotFoundError(f"숫자로 된 하위 폴더가 없습니다: {input_dir}")
    folder_options = [d.name for d in subdirs]
    print(f"Select Folder: {folder_options}")
    # 하드코딩된 선택 (예: "11081")
    selected_folder_name = "11081"
    selected_folder = input_dir / selected_folder_name
    json_files = list(selected_folder.glob(f'*_standard_form_{selected_folder_name}.json'))
    if not json_files:
        raise FileNotFoundError(f"선택된 폴더에 JSON 파일이 없습니다: {selected_folder}")
    print(f"처리 중인 폴더: {selected_folder_name}")
    print(f"처리할 JSON 파일 수: {len(json_files)}")
    
    # 출력 폴더: 현재 train 폴더 내에 생성 (예: Input_for_kcBERT_{selected_folder_name})
    output_dir = Path.cwd() / f'Input_for_kcBERT_{selected_folder_name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # BERT 토크나이저 초기화 (Fast 버전)
    tokenizer = BertTokenizerFast.from_pretrained('beomi/kcbert-base')
    
    output_file = output_dir / f'Input_for_kcBERT_{selected_folder_name}.tsv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 허용 라벨 및 최대 포함 개수
    ALLOWED_LABELS = {"이름", "URL", "소속", "주소", "신원", "계정", "번호", "금융"}
    label_counts = {
        "이름": 10000,
        "URL": 3000,
        "소속": 10000,
        "주소": 6427,
        "신원": 703,
        "계정": 3629,
        "번호": 4022,
        "금융": 2162,
        "None_Dialogues": 100
    }
    NEAR_LABEL_CHARS = 120  # 라벨 전후 포함할 문자 수
    MIN_TOKENS = 30        # 시퀀스 최소 토큰 수
    
    included_label_counts = {label: 0 for label in ALLOWED_LABELS}
    included_none_dialogues = 0
    
    def is_all_labels_reached_max():
        for label in ALLOWED_LABELS:
            if included_label_counts[label] < label_counts[label]:
                return False
        if included_none_dialogues < label_counts["None_Dialogues"]:
            return False
        return True
    
    sentences = []
    labels_list = []
    
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for json_file in tqdm(json_files, desc="파일 처리 중", unit="file"):
            if is_all_labels_reached_max():
                print("모든 라벨의 최대 개수에 도달하여 처리를 종료합니다.")
                break
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except json.JSONDecodeError:
                print(f"JSON 디코딩 오류: 파일 {json_file.name} 건너뜀")
                continue
            dialogues = json_data.get('data', [])
            if not dialogues:
                continue
            for dialogue in dialogues:
                dialogue_label_counts = {label: 0 for label in ALLOWED_LABELS}
                utterances_data = dialogue.get('body', [])
                for utterance_data in utterances_data:
                    labels_in_utterance = utterance_data.get('labels', [])
                    labels_in_utterance = [label for label in labels_in_utterance if label in ALLOWED_LABELS]
                    for label in labels_in_utterance:
                        dialogue_label_counts[label] += 1
                total_labels_in_dialogue = sum(dialogue_label_counts.values())
                include_dialogue = False
                if total_labels_in_dialogue == 0:
                    if included_none_dialogues < label_counts["None_Dialogues"]:
                        include_dialogue = True
                        nonlocal_none = 1  # dummy update
                        included_none_dialogues += 1
                    else:
                        include_dialogue = False
                else:
                    for label in ALLOWED_LABELS:
                        if included_label_counts[label] < label_counts[label] and dialogue_label_counts[label] > 0:
                            include_dialogue = True
                            break
                if not include_dialogue:
                    continue
                for label in ALLOWED_LABELS:
                    included_label_counts[label] += dialogue_label_counts[label]
                
                dialogue_char_position = 0
                label_positions_in_dialogue = []
                utterance_positions_in_dialogue = []
                for utterance_data in utterances_data:
                    utterance_text = utterance_data.get('utterance', '')
                    labels_in_utterance = utterance_data.get('labels', [])
                    label_positions = utterance_data.get('labelPositions', [])
                    labels_positions = [(label, position) for label, position in zip(labels_in_utterance, label_positions) if label in ALLOWED_LABELS]
                    utterance_start = dialogue_char_position
                    utterance_end = dialogue_char_position + len(utterance_text)
                    for label, position in labels_positions:
                        try:
                            start_pos, end_pos = map(int, position.split('-'))
                        except ValueError:
                            print(f"잘못된 위치 형식: {position} in file {json_file.name}")
                            continue
                        label_start = utterance_start + start_pos
                        label_end = utterance_start + end_pos
                        label_positions_in_dialogue.append((label_start, label_end))
                    utterance_positions_in_dialogue.append((utterance_data, utterance_start, utterance_end))
                    dialogue_char_position = utterance_end
                if total_labels_in_dialogue > 0:
                    included_utterances = []
                    for utterance_data, utterance_start, utterance_end in utterance_positions_in_dialogue:
                        include_utterance = False
                        for label_start, label_end in label_positions_in_dialogue:
                            if utterance_end >= label_start - NEAR_LABEL_CHARS and utterance_start <= label_end + NEAR_LABEL_CHARS:
                                include_utterance = True
                                break
                        if include_utterance:
                            included_utterances.append(utterance_data)
                else:
                    included_utterances = utterances_data
                utterances_tokens_labels = []
                for utterance_data in included_utterances:
                    utterance = utterance_data.get('utterance', '')
                    labels_in_utterance = utterance_data.get('labels', [])
                    label_positions = utterance_data.get('labelPositions', [])
                    labels_positions = [(label, position) for label, position in zip(labels_in_utterance, label_positions) if label in ALLOWED_LABELS]
                    tokens = tokenizer.tokenize(utterance)
                    token_positions = []
                    char_pointer = 0
                    for token in tokens:
                        clean_token = token.replace('##', '')
                        token_start = utterance.find(clean_token, char_pointer)
                        if token_start == -1:
                            token_start = char_pointer
                        token_end = token_start + len(clean_token)
                        token_positions.append((token_start, token_end))
                        char_pointer = token_end
                    token_labels = ['O'] * len(tokens)
                    for label, position in labels_positions:
                        try:
                            start_pos, end_pos = map(int, position.split('-'))
                        except ValueError:
                            print(f"잘못된 위치 형식: {position} in file {json_file.name}")
                            continue
                        entity_tokens = []
                        for idx_token, (token_start, token_end) in enumerate(token_positions):
                            if token_end <= start_pos:
                                continue
                            if token_start >= end_pos:
                                break
                            if token_start < end_pos and token_end > start_pos:
                                entity_tokens.append(idx_token)
                        for idx, idx_token in enumerate(entity_tokens):
                            if token_labels[idx_token] != 'O':
                                continue
                            if idx == 0:
                                token_labels[idx_token] = f'B-{label}'
                            else:
                                token_labels[idx_token] = f'I-{label}'
                    if len(tokens) > 300:
                        chunks = sliding_window_split(tokens, token_labels, max_length=300, stride=50)
                        for chunk_tokens, chunk_labels in chunks:
                            if len(chunk_tokens) >= MIN_TOKENS:
                                for token, label in zip(chunk_tokens, chunk_labels):
                                    writer.writerow([token, label])
                                writer.writerow([])
                                sentences.append(' '.join(chunk_tokens))
                                labels_list.append(chunk_labels)
                    else:
                        utterances_tokens_labels.append((tokens, token_labels))
                        sentences.append(' '.join(tokens))
                        labels_list.append(token_labels)
                num_utterances = len(utterances_tokens_labels)
                w_start = 0
                while w_start < num_utterances:
                    current_tokens = []
                    current_labels = []
                    w_end = w_start
                    while w_end < num_utterances and len(current_tokens) + len(utterances_tokens_labels[w_end][0]) <= 300:
                        utter_tokens, utter_labels = utterances_tokens_labels[w_end]
                        current_tokens.extend(utter_tokens)
                        current_labels.extend(utter_labels)
                        w_end += 1
                    if len(current_tokens) >= MIN_TOKENS:
                        for token, label in zip(current_tokens, current_labels):
                            writer.writerow([token, label])
                        writer.writerow([])
                        sentences.append(' '.join(current_tokens))
                        labels_list.append(current_labels)
                    if w_end == w_start:
                        w_start += 1
                    else:
                        w_start = w_end
        # End for each json_file
    print(f"데이터가 '{output_file}'에 저장되었습니다.")
    print("포함된 라벨 개수:")
    for label in ALLOWED_LABELS:
        print(f"{label}: {included_label_counts[label]}")
    print(f"라벨이 없는 대화 포함 개수: {included_none_dialogues}")
    
    sentences_labels_file = output_dir / f'sentences_labels_{selected_folder_name}.pkl'
    with open(sentences_labels_file, 'wb') as f:
        pickle.dump({'sentences': sentences, 'labels': labels_list}, f)
    print(f"원본 문장과 라벨이 '{sentences_labels_file}'에 저장되었습니다.")

###########################################
# Main 함수: Part A와 Part B 순차 실행
###########################################
def main():
    print("=== Part A: Standard Form 생성 시작 ===")
    std_form_dir = generate_standard_form()
    if std_form_dir is None:
        print("Standard Form 생성 실패. 종료합니다.")
        return
    print("=== Part A 완료 ===\n")
    
    print("=== Part B: BERT 학습 데이터 생성 시작 ===")
    generate_bert_input()
    print("=== Part B 완료 ===")

if __name__ == "__main__":
    main()
