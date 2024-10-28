import os
import json
from collections import defaultdict

file_path = './nbest'  # nbest_predictions.json 파일들이 있는 경로
n = 10  # 고려할 상위 예측 개수

json_files = []

# 모든 JSON 파일 경로 수집
for json_filename in os.listdir(file_path):
    if json_filename.endswith('.json'):
        json_files.append(os.path.join(file_path, json_filename))

# 파일에서 JSON 데이터 로드
nbest_data = []
for file in json_files:
    try:
        with open(file, "r", encoding='utf-8') as json_file:
            nbest_data.append(json.load(json_file))
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading {file}: {e}")

# 확률 및 최종 답변 저장을 위한 딕셔너리 초기화
probability_bins = defaultdict(list)
final_answers = {}

# 모든 JSON 파일에서 확률 집계
for data in nbest_data:
    for key in data.keys():
        probability_bins[key].extend(
            [(entry['text'], entry['probability']) for entry in data[key][:n]]
        )

# 최대 확률에 기반하여 최적의 답변 결정
for key, entries in probability_bins.items():
    answer_hubo = defaultdict(float)
    for text, prob in entries:
        answer_hubo[text] += prob  # 확률 누적

    # 가장 높은 누적 확률을 가진 답변 선택
    final_answers[key] = max(answer_hubo, key=answer_hubo.get)

# 최종 앙상블 결과를 새로운 JSON 파일에 저장
new_nbest_json_path = "./ensemble/predictions.json"
with open(new_nbest_json_path, 'w', encoding='utf-8') as file:
    json.dump(final_answers, file, ensure_ascii=False, indent=4)