# Korean-PII-Masking-BERT
한국어 데이터 개인정보 마스킹 BERT

---
Korean-PII-Masking-BERT는 한국어 개인정보 데이터를 보호하기 위해 BERT 기반의 마스킹 및 가명정보 생성 파이프라인을 제공합니다.  
이 저장소는 다음과 같은 주요 기능을 포함합니다.

- **데이터 전처리**: 원본 SNS JSON 데이터를 정제하고, 불필요한 정보를 제거하여 표준 형식의 JSON 파일로 변환합니다.
- **가명정보 생성**: 로컬 규칙과 GPT API를 활용하여 민감한 정보를 가명으로 대체합니다.
- **BERT 파인튜닝**: KcBERT 모델에 CRF 레이어를 추가하여 토큰 분류 성능을 향상시킵니다.
- **엔드 투 엔드 자동화**: 각 단계별 스크립트를 개별적으로 실행할 수 있으며, `app.py` 스크립트를 통해 전체 파이프라인을 순차적으로 실행할 수 있습니다.

## 디렉토리 구조
Korean-PII-Masking-BERT/
├── data_generation/
│   ├── app.py                           # 전체 실행 (preprocess.py, prepare_batch_input.py, process_gpt_api_responses.py)
│   ├── preprocess.py                    # 원본 JSON 파일 전처리
│   ├── prepare_batch_input.py           # GPT API 배치 입력 생성 (가명정보 생성 전 단계)
│   ├── process_gpt_api_responses.py     # GPT API 응답 통합 후 최종 가명정보 JSON 생성
│   └── data/
│       ├── fake_pii/                    # CSV 파일들 (원본은 공개하지 않음)
│       ├── Korean_SNS_final_preprocessed/       # 전처리 완료 JSON 파일 (preprocess.py 산출)
│       ├── Korean_SNS_intermediate_generated/   # 중간 가명정보 생성 산출물 (prepare_batch_input.py 산출)
│       └── Korean_SNS_final_generated/          # 최종 가명정보 JSON 파일 (process_gpt_api_responses.py 산출)
├── llm_input/
│   ├── extracted_dialogues/             # LLM 입력 데이터 (JSONL)
│   └── batchapi/                        # GPT API 배치 입력 파일
├── llm_output/
│   └── batchapi/                        # GPT API 응답 배치 파일 (JSONL)
└── train/
    ├── app.py                           # 전체 실행 (preprocess.py, KcBERT+CRF_finetune.py)
    ├── preprocess.py                    # 최종 가명정보 JSON을 표준 형식으로 변환하여 학습 데이터 생성
    ├── KcBERT+CRF_finetune.py             # KcBERT + CRF 모델 파인튜닝
    └── data/
        └── standard_form/               # 표준 형식의 JSON 파일 및 학습 데이터 저장

