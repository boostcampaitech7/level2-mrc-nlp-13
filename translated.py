from datasets import Dataset

base_train = Dataset.load_from_disk('/data/ephemeral/home/sungeun/level2-mrc-nlp-13/data/train_dataset/train')
base_train_df = base_train.to_pandas()

from googletrans import Translator
import time

# Translator 객체 생성
translator = Translator()

def back_translate(text, src='ko', mid='ja', dest='ko'):
    """
    역번역 함수: 한국어 -> 일본어 -> 한국어
    src: 원본 언어 (한국어)
    mid: 중간 번역 언어 (일본어)
    dest: 최종 언어 (한국어)
    """
    try:
        # 한국어에서 일본어로 번역
        translated_to_mid = translator.translate(text, src=src, dest=mid).text
        time.sleep(1)  # 너무 많은 요청을 한 번에 보내지 않도록 대기
        # 일본어에서 다시 한국어로 번역
        back_translated = translator.translate(translated_to_mid, src=mid, dest=dest).text
        time.sleep(1)  # 대기
        return back_translated
    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # 오류 발생 시 원본 텍스트 반환

# 역번역 적용하기
for idx, row in base_train_df.iterrows():
    base_train_df.at[idx, 'context'] = back_translate(row['context'])
    base_train_df.at[idx, 'question'] = back_translate(row['question'])
    # base_train_df.at[idx, 'answers']['text'] = [back_translate(answer) for answer in row['answers']['text']]

    if idx % 10 == 0:  # 매 10번째 행마다 상태 출력
        print(f"{idx} rows processed...")