import re
from datasets import load_from_disk
import json

def clean_text(text):
    # 정규 표현식 패턴 정의
    newline_pattern = r'\n'

    # 정규 표현식을 사용하여 패턴 제거
    text = re.sub(newline_pattern, ' ', text)
    return text.strip()

# Dataset 로드
train_dataset = load_from_disk("../data/train_dataset/train")
dev_dataset = load_from_disk("../data/train_dataset/validation")
# 'context' 열의 텍스트 정제
cleaned_contexts = [clean_text(context) for context in train_dataset['context']]
cleaned_contexts_ = [clean_text(context) for context in dev_dataset['context']]

# 정제된 텍스트로 'context' 열 업데이트
train_dataset = train_dataset.remove_columns(['context'])
dev_dataset = dev_dataset.remove_columns(['context'])

train_dataset = train_dataset.add_column('context', cleaned_contexts)
dev_dataset = dev_dataset.add_column('context', cleaned_contexts_)
# 수정된 Dataset을 원래 위치에 저장 (덮어쓰기)

train_dataset.save_to_disk("../data/train_dataset_clean/train")
dev_dataset.save_to_disk("../data/train_dataset_clean/validation")

print("Dataset has been cleaned and saved successfully.")

with open("../data/wikipedia_documents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 각 문서의 'text' 필드에서 패턴 제거
for doc_id, doc_info in data.items():
    if 'text' in doc_info:
        doc_info['text'] = clean_text(doc_info['text'])

# 수정된 데이터를 새 JSON 파일로 저장
with open("../data/wikipedia_documents_clean.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

print("Wikipedia documents have been cleaned and saved successfully.")