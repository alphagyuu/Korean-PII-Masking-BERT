# train/train_bert_crf.py

import tensorflow as tf
import numpy as np
import os
import pandas as pd
from transformers import BertTokenizer, TFBertForTokenClassification
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
from sklearn.utils import class_weight
import matplotlib.pyplot as plt  # 시각화 관련 함수는 사용하지 않으므로 제거 가능

# ------------------------------
# 전역 변수 및 실험 변수 설정
# ------------------------------
version = "11082"  # 버전 번호 (실험 버전을 쉽게 변경 가능)
max_len = 512      # 최대 시퀀스 길이
batch_size = 8     # 배치 사이즈
random_state = 2018  # train/val 분할 시 사용

# ------------------------------
# 디렉토리 설정 (train 폴더 내에서 실행한다고 가정)
# ------------------------------
BASE_DIR = Path.cwd()  # 현재 train 폴더 (예: Korean-PII-Masking-BERT/train)
input_dir = BASE_DIR / f"Input_for_kcBERT_{version}"
crf_dir = BASE_DIR / f"CRF_{version}"   # CRF 관련 임시 데이터는 train 폴더 내에 저장
checkpoint_dir = input_dir / "CHECKPOINTS"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
data_file = input_dir / f"Input_for_kcBERT_{version}.tsv"

# 최종 모델 저장 경로 설정
project_root = BASE_DIR.parent  # train 폴더의 상위: Korean-PII-Masking-BERT
saved_model_dir = project_root / "models" / "Korean-PII-Masking-BertForTokenClassification"
saved_model_dir.mkdir(parents=True, exist_ok=True)

# CRF 모델 저장 경로
crf_save_dir = project_root / "models" / "Korean-PII-Masking-CRF"
crf_save_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------
# 1. 데이터 로드 및 문장/태그 그룹화
# ------------------------------
sentences = []
labels = []
with open(data_file, 'r', encoding='utf-8') as f:
    sentence = []
    label = []
    for line in f:
        line = line.strip()
        if line == '':
            if len(sentence) > 0:
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
        else:
            splits = line.split('\t')
            if len(splits) >= 2:
                word = splits[0]
                tag = splits[1]
                sentence.append(word)
                label.append(tag)
    if len(sentence) > 0:
        sentences.append(sentence)
        labels.append(label)

print(f"총 문장 수: {len(sentences)}")

# ------------------------------
# 2. 태그 정렬 및 사전 생성
# ------------------------------
def sort_tags(tags):
    tags = list(tags)
    if "O" in tags:
        tags.remove("O")
    sorted_tags = sorted(tags, key=lambda x: (x[2:] if '-' in x else x, x[0]))
    return ["O"] + sorted_tags

words = list(set([w for s in sentences for w in s]))
tags = list(set([t for l in labels for t in l]))
tags = sort_tags(tags)
print(f"정렬된 태그 목록: {tags}")

tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

# ------------------------------
# 3. 토크나이저 초기화 및 토큰화 (레이블 보존)
# ------------------------------
tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base', do_lower_case=False)

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels_out = []
    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        tokenized_sentence.extend(tokenized_word)
        labels_out.extend([label] * len(tokenized_word))
    return tokenized_sentence, labels_out

tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs)
                              for sent, labs in zip(sentences, labels)]
tokenized_texts = [pair[0] for pair in tokenized_texts_and_labels]
labels_tokenized = [pair[1] for pair in tokenized_texts_and_labels]

# ------------------------------
# 4. 인덱스 변환 및 패딩
# ------------------------------
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype="long", value=0,
                          truncating="post", padding="post")

if "O" not in tag2idx:
    tag2idx["O"] = len(tag2idx)
    idx2tag[len(tag2idx)-1] = "O"

tags_padded = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels_tokenized],
                           maxlen=max_len, value=tag2idx["O"], padding="post",
                           dtype="long", truncating="post")

# ------------------------------
# 5. 어텐션 마스크 생성 및 학습/검증 데이터 분할
# ------------------------------
attention_masks = [[float(i != 0) for i in seq] for seq in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags_padded,
                                                            random_state=random_state, test_size=0.1)
tr_masks, val_masks = train_test_split(attention_masks,
                                       random_state=random_state, test_size=0.1)

tr_inputs = tf.convert_to_tensor(tr_inputs)
val_inputs = tf.convert_to_tensor(val_inputs)
tr_tags = tf.convert_to_tensor(tr_tags)
val_tags = tf.convert_to_tensor(val_tags)
tr_masks = tf.convert_to_tensor(tr_masks)
val_masks = tf.convert_to_tensor(val_masks)

# ------------------------------
# 6. 클래스 가중치 계산
# ------------------------------
train_labels = tr_tags.numpy().flatten()
classes = np.unique(train_labels)
class_counts = np.bincount(train_labels)
print(f"클래스 카운트: {class_counts}")

beta = 0.999
effective_num = 1.0 - np.power(beta, class_counts)
class_weights = (1.0 - beta) / effective_num
class_weights = class_weights / np.mean(class_weights)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
class_weights_dict[0] = 0.6
print(f"조정된 클래스 가중치: {class_weights_dict}")

class_weights_array = np.zeros(len(tag2idx))
for idx in range(len(tag2idx)):
    class_weights_array[idx] = class_weights_dict[idx]

def create_sample_weight(tags, class_weights):
    return np.take(class_weights, tags)

sample_weights = create_sample_weight(tr_tags.numpy(), class_weights_array)

# ------------------------------
# 7. 모델 초기화 및 학습
# ------------------------------
model = TFBertForTokenClassification.from_pretrained('beomi/kcbert-base', num_labels=len(tag2idx))

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric], sample_weight_mode='temporal')

# 체크포인트 디렉토리 설정 (train/CHECKPOINTS)
checkpoint_dir = BASE_DIR / "CHECKPOINTS"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f"최신 체크포인트 {ckpt_manager.latest_checkpoint}에서 모델을 복원합니다.")
    initial_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
else:
    print("체크포인트가 없어 모델을 처음부터 학습합니다.")
    initial_epoch = 0

class CustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_manager):
        self.ckpt_manager = ckpt_manager
    def on_epoch_end(self, epoch, logs=None):
        ckpt.step.assign_add(1)
        save_path = self.ckpt_manager.save()
        print(f"에포크 {epoch+1}에서 체크포인트 저장: {save_path}")

checkpoint_callback = CustomCheckpoint(ckpt_manager)

history = model.fit(
    [tr_inputs, tr_masks],
    tr_tags,
    sample_weight=sample_weights,
    validation_data=([val_inputs, val_masks], val_tags),
    epochs=5,
    batch_size=batch_size,
    callbacks=[checkpoint_callback],
    initial_epoch=initial_epoch
)

# ------------------------------
# 8. 모델 및 토크나이저 저장
# ------------------------------
model.save_pretrained(saved_model_dir)
tokenizer.save_pretrained(saved_model_dir)
print(f"모델과 토크나이저가 {saved_model_dir}에 저장되었습니다.")

# ------------------------------
# 9. CRF 데이터셋 생성 및 저장 (NumPy .npz 파일)
# ------------------------------
print("KcBERT 모델 추론 시작 (train 데이터)...")
train_logits = model.predict([tr_inputs.numpy(), tr_masks.numpy()]).logits
print("KcBERT 모델 추론 완료 (train 데이터).")

print("KcBERT 모델 추론 시작 (val 데이터)...")
val_logits = model.predict([val_inputs.numpy(), val_masks.numpy()]).logits
print("KcBERT 모델 추론 완료 (val 데이터).")

if not os.path.exists(crf_dir):
    os.makedirs(crf_dir)

np.savez(os.path.join(crf_dir, f'train_crf.npz'),
         emissions=train_logits, labels=tr_tags.numpy(), mask=tr_masks.numpy())
np.savez(os.path.join(crf_dir, f'val_crf.npz'),
         emissions=val_logits, labels=val_tags.numpy(), mask=val_masks.numpy())

print(f"CRF 학습용 데이터셋이 {crf_dir}에 저장되었습니다.")

# ------------------------------
# 10. PyTorch CRF 모델 학습 및 저장
# ------------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from tqdm import tqdm

class CRFDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.emissions = data['emissions']
        self.labels = data['labels']
        self.mask = data['mask']

    def __len__(self):
        return len(self.emissions)

    def __getitem__(self, idx):
        emissions = torch.tensor(self.emissions[idx], dtype=torch.float)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        mask = torch.tensor(self.mask[idx], dtype=torch.bool)
        return emissions, labels, mask

train_npz_path = os.path.join(crf_dir, f'train_crf.npz')
val_npz_path = os.path.join(crf_dir, f'val_crf.npz')

train_dataset = CRFDataset(train_npz_path)
val_dataset = CRFDataset(val_npz_path)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

sample_emission, _, _ = train_dataset[0]
input_dim = sample_emission.shape[-1]
num_tags = len(tag2idx)

class CRFModel(nn.Module):
    def __init__(self, input_dim, num_tags):
        super(CRFModel, self).__init__()
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, emissions, tags, mask):
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss

    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)

model_crf = CRFModel(input_dim, num_tags)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_crf.to(device)

optimizer = torch.optim.Adam(model_crf.parameters(), lr=1e-3)
num_epochs = 20
patience = 3
best_val_loss = float('inf')
early_stop_counter = 0

print("PyTorch CRF 모델 학습 시작...")

for epoch in range(num_epochs):
    model_crf.train()
    total_loss = 0.0
    for emissions, labels, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit="batch"):
        emissions = emissions.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        loss = model_crf(emissions, labels, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")
    
    model_crf.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for emissions, labels, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", unit="batch"):
            emissions = emissions.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            loss = model_crf(emissions, labels, mask)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        best_model_state = model_crf.state_dict()
    else:
        early_stop_counter += 1
        print(f"EarlyStopping 카운터: {early_stop_counter} / {patience}")
        if early_stop_counter >= patience:
            print("Early Stopping 조건 만족 - 학습 중단합니다.")
            break

print("PyTorch CRF 모델 학습 완료.")

crf_model_path = os.path.join(crf_save_dir, f'crf_model_{version}.pt')
torch.save(best_model_state, crf_model_path)
print(f"CRF 모델이 {crf_model_path}에 저장되었습니다.")

print("CRF 모델 추론 시작 (validation 데이터)...")
model_crf.eval()
all_predictions = []
with torch.no_grad():
    for emissions, labels, mask in tqdm(val_loader, desc="CRF Inference", unit="batch"):
        emissions = emissions.to(device)
        mask = mask.to(device)
        predictions = model_crf.decode(emissions, mask)
        all_predictions.extend(predictions)
print("CRF 모델 추론 완료.")
