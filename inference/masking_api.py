import os
import tensorflow as tf
import torch
import numpy as np
from transformers import TFBertForTokenClassification, BertTokenizerFast
from huggingface_hub import hf_hub_download
from torch import nn
from torchcrf import CRF
from fastapi import FastAPI
from pydantic import BaseModel

# =============================================================================
# 1. CRF 모델 정의 (PyTorch)
# =============================================================================
class CRFModel(nn.Module):
    def __init__(self, num_tags):
        super(CRFModel, self).__init__()
        self.crf = CRF(num_tags, batch_first=True)
        
    def forward(self, emissions, tags, mask):
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss
    
    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)

# =============================================================================
# 2. 모델 및 토크나이저 불러오기
# =============================================================================
model_name = "alphagyuu/Korean-PII-Masking-BertForTokenClassification"
tf_model = TFBertForTokenClassification.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# 정렬된 태그 목록 기반 id2label 재정의
sorted_tags = [
    'O', 'B-URL', 'I-URL', 'B-계정', 'I-계정', 'B-금융', 'I-금융', 
    'B-번호', 'I-번호', 'B-소속', 'I-소속', 'B-신원', 'I-신원', 
    'B-이름', 'I-이름', 'B-주소', 'I-주소'
]
id2label = {i: tag for i, tag in enumerate(sorted_tags)}
num_tags = len(sorted_tags)

# =============================================================================
# 3. CRF 모델 불러오기 (PyTorch)
# =============================================================================
crf_repo = "alphagyuu/Korean-PII-Masking-CRF"
crf_filename = "crf_model_11082.pt"
crf_model_file = hf_hub_download(repo_id=crf_repo, filename=crf_filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crf_model = CRFModel(num_tags)
state_dict = torch.load(crf_model_file, map_location=device)
crf_model.load_state_dict(state_dict)
crf_model.to(device)
crf_model.eval()

# =============================================================================
# 4. BERT+CRF 추론 함수 (다수결로 엔티티 라벨 결정)
# =============================================================================
def predict_entities_crf(text: str, max_length: int = 512):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="tf",
        truncation=True,
        max_length=max_length
    )
    attention_mask = encoding["attention_mask"].numpy()[0]
    offsets = encoding.pop("offset_mapping").numpy()[0].tolist()
    outputs = tf_model(encoding)
    logits_np = outputs.logits.numpy()
    emissions = torch.tensor(logits_np, dtype=torch.float).to(device)
    attn_mask = torch.tensor(attention_mask, dtype=torch.bool).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_indices = crf_model.decode(emissions, mask=attn_mask)[0]
    pred_tags = [id2label.get(idx, "O") for idx in pred_indices]
    
    entities = []
    current_entity_start = None
    current_entity_end = None
    current_labels = []
    for tag, offset in zip(pred_tags, offsets):
        start_char, end_char = offset
        if start_char == 0 and end_char == 0:
            continue
        if tag.startswith("B-"):
            if current_entity_start is not None:
                # 결정: 다수결로 최종 라벨 결정
                majority_label = max(set(current_labels), key=current_labels.count)
                entities.append((current_entity_start, current_entity_end, majority_label))
            current_entity_start = start_char
            current_entity_end = end_char
            current_labels = [tag.split("-", maxsplit=1)[1]]
        elif tag.startswith("I-"):
            if current_entity_start is not None:
                current_entity_end = end_char
                current_labels.append(tag.split("-", maxsplit=1)[1])
            else:
                continue
        else:  # "O"인 경우
            if current_entity_start is not None:
                majority_label = max(set(current_labels), key=current_labels.count)
                entities.append((current_entity_start, current_entity_end, majority_label))
                current_entity_start = None
                current_entity_end = None
                current_labels = []
    if current_entity_start is not None:
        majority_label = max(set(current_labels), key=current_labels.count)
        entities.append((current_entity_start, current_entity_end, majority_label))
    return entities

# =============================================================================
# 5. 마스킹 및 위치정보 함수
# =============================================================================
def mask_text_crf(text: str) -> str:
    entities = predict_entities_crf(text)
    if not entities:
        return text
    entities = sorted(entities, key=lambda x: x[0])
    masked_text = ""
    last_idx = 0
    for start, end, label in entities:
        masked_text += text[last_idx:start] + f"#@{label}#"
        last_idx = end
    masked_text += text[last_idx:]
    return masked_text

def get_entity_locations_crf(text: str) -> dict:
    entities = predict_entities_crf(text)
    locations = {}
    for start, end, label in entities:
        pos_str = f"{start}-{end}"
        locations.setdefault(label, []).append(pos_str)
    return locations

# =============================================================================
# 6. FastAPI 로컬 API 구현
# =============================================================================
app = FastAPI(title="Korean PII Masking API (BERT+CRF)")

class TextRequest(BaseModel):
    text: str

@app.post("/mask")
def mask_text_endpoint(req: TextRequest):
    masked = mask_text_crf(req.text)
    return {"masked_text": masked}

@app.post("/locations")
def get_locations_endpoint(req: TextRequest):
    locs = get_entity_locations_crf(req.text)
    return {"locations": locs}
