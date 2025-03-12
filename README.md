# Korean-PII-Masking-BERT  
한국어 데이터 개인정보 마스킹 BERT


## Hugging Face Model Card 😊
- [alphagyuu/Korean-PII-Masking-BertForTokenClassification](https://huggingface.co/alphagyuu/Korean-PII-Masking-BertForTokenClassification)


---

Korean-PII-Masking-BERT는 가명정보를 통해 가공한 한국어 SNS 데이터셋을 기반으로 [KcBERT](https://github.com/Beomi/KcBERT)를 파인튜닝하고, CRF Layer를 결합한 토큰 분류 모델입니다.  
상세한 내용은 "[생성형AI 시대의 한국어 데이터를 위한 개인정보 보호: KcBERT와 Chain-of-Thought 프롬프팅 기반 하이브리드 접근을 중심으로](https://www.earticle.net/Article/A463753)" 논문에 자세히 소개되어 있습니다.


## Introduction 💡
Korean-PII-Masking-BERT는 가명정보 생성 파이프라인과 KcBERT 기반 토큰 분류 모델을 결합한 시스템으로, 학습 및 추론 전 과정을 공개합니다.


## Features ✨
- **데이터 전처리**  
  한국어 SNS JSON 데이터를 정제하고 불필요한 정보를 제거하여 표준 형식의 JSON 파일로 변환합니다.
- **가명정보 생성**  
  gpt-4o-mini 모델과 간단한 Chain-of-Thought 기법을 적용하여 민감 정보를 가명으로 대체합니다. 일부 정형 정보는 사전에 수집한 가명정보 데이터셋으로부터 생성됩니다.
- **BERT 파인튜닝**  
  KcBERT 모델에 CRF Layer를 추가하여 토큰 분류 성능을 향상시킵니다.
- **파이프라인 자동화**  
  각 단계별 스크립트를 개별적으로 실행할 수 있으며, `app.py`를 통해 전체 파이프라인을 순차적으로 실행할 수 있습니다.


## Directory Structure 📁
```plaintext
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
```


## How to Use ▶️
1. 각 단계별 디렉토리의 `app.py` 파일을 실행하여 전체 파이프라인을 순차적으로 구동합니다.
2. 데이터 전처리, 가명정보 생성, 모델 파인튜닝 등 필요한 단계를 개별적으로 실행할 수 있습니다.
3. 최종 생성된 모델은 학습된 토큰 분류 작업에 활용되며, 실제 서비스 환경에 쉽게 통합할 수 있습니다.


## Evaluation 📊
가명정보 생성과 동일한 파이프라인으로 독립적으로 생성된 평가용 데이터셋에 대해,  
- **Macro-average Precision**: 0.96  
- **Recall**: 0.91  
- **F1-Score**: 0.94  
를 기록하였습니다.


## Limitations ⚠️
- 짧은 문장에서는 context 부족으로 masking rate가 저하되는 현상이 있어 추가 학습이 필요합니다.
- 실사용 시 100% 마스킹이 보장되지 않아 주의해야 합니다.


## Funding 💰
본 결과물은 교육부와 한국연구재단의 재원으로 지원을 받아 수행된 디지털 신기술 인재양성 혁신공유대학사업의 연구결과입니다.  
이 연구는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 "데이터명"을 활용하여 수행된 연구입니다.  
본 연구에 활용된 데이터는 [AI 허브](https://aihub.or.kr)에서 다운로드 받으실 수 있습니다.  
This research (paper) used datasets from 'The Open AI Dataset Project (AI-Hub, S. Korea)'.  
All data information can be accessed through [AI-Hub](https://www.aihub.or.kr).


## License 📜
apache-2.0 [LICENSE](LICENSE) 
