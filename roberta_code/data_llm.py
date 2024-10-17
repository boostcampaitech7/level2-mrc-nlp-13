# import openai
# import os

# # 환경 변수에서 API 키 로드 (또는 API 키를 직접 입력)
# openai.api_key = os.getenv("OPENAI_API_KEY",
#                            "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")

# # GPT-4 모델을 사용해 텍스트 생성하는 함수
# def generate_gpt4_context(prompt, model="gpt-4", max_tokens=150):
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt},
#         ],
#         max_tokens=max_tokens,
#         temperature=0.7,  # 더 높은 값은 더 창의적인 결과를, 낮은 값은 더 보수적인 결과를 생성
#     )
#     return response['choices'][0]['message']['content']

# # 예시 프롬프트
# prompt = "Explain how transformers work in natural language processing."

# # GPT-4를 사용하여 응답 생성
# response = generate_gpt4_context(prompt)
# print("GPT-4 Response:", response)

from transformers import BartTokenizer, BartForConditionalGeneration

# KoBART 모델과 토크나이저 로드
tokenizer = BartTokenizer.from_pretrained("gogamza/kobart-base-v2")
model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")

# 긴 문맥 입력
context = """
한국의 역사와 문화는 매우 오래된 전통을 가지고 있습니다. 특히 조선 시대는 예술과 학문이 번성했던 시기였습니다. 이 시기 동안, 성리학이 중심 사상으로 자리 잡았고, 한글이 창제되었으며, 다양한 과학 기술도 발전했습니다.
"""

# 질문 생성 프롬프트 준비
input_text = f"Context: {context}\nGenerate a question based on the context."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 질문 생성
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
question = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(question)
