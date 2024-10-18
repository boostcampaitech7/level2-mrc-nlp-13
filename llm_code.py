'''
import torch
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel

from datasets import Dataset


# load file
base_train = Dataset.load_from_disk('/data/ephemeral/home/sungeun/level2-mrc-nlp-13/data/train_dataset/train')
base_train_df = base_train.to_pandas()

# 'title'이 '윤치호'인 행을 먼저 필터링한 후 중복된 'context' 제거
yunchiho_unique_context_rows = base_train_df[base_train_df['title'] == '윤치호']

grouped = yunchiho_unique_context_rows.groupby('context')['question'].apply(list).reset_index()
context = grouped['context'][0]
# '1884년 1월 18일부터 8월 9일까지 윤치호는 거듭하여 사관학교 설립을 상주한다. 윤치호는 군대 통솔권의 일원화 군인정신의 합일, 상무정신의 강화를 통하여 충성스럽고 용감한 국방군을 양성해야 한다고 보았던 것이다. 이러한 목적에서 그는 미국인 군사교관을 초빙하여 각 영을 통합훈련할 것 과 사관학교 설립을 건의했던 것이다. 이어 병원과 학교의 설립 및 전신국의 설치를 미국인에게 허가해줄 것을 건의하는 등 근대시설의 도입에도 깊은 관심을 보였다\\n\\n1884년 7월에는 선교사들을 통해 신식 병원과 전화국을 유치, 개설할 것을 고종에게 상주하여 허락받았다. 그러나 신식 병원 도입과 전화국 개통은 갑신정변의 실패로 전면 백지화된다. 1894년 9월 무렵 그는 일본의 조선 침략을 예상하였다. \'일본은 이제까지는 개혁을 조선인 스스로 하도록 하려 했다. 그러나 그들이 볼 때 조선인들이 개혁의 의욕도 능력도 없음을 보고 주도권을 잡기로 결심한 것 같다 \'며 일본이 한국에 영향력을 행사하리라고 전망했다. 1884년 12월의 갑신정변 직전까지 그는 온건파 개화당의 일원으로 자주독립과 참정권, 부국강병을 위해 활동하였다. 영어 실력의 부족함을 느낀 그는 다시 주조선미국 공사관의 직원들과 교류하며 자신에게 영어를 가르쳐줄 것을 부탁하여 주조선미국공사관 직원 미군 중위 존 B. 베르나든(John B. Bernadon)이 이를 수락하였다. 5월 그는 1개월간 베르나든에게 하루 한 시간씩 영어 개인 지도를 받기도 했다.\\n\\n1884년 12월 갑신정변 초기에 윤치호는 정변 계획을 접하고 혁명의 성공을 기대하였다. 당시 김옥균을 믿고 따랐던 그는 1894년 11월에 접어들면서 윤치호는 아버지인 윤웅렬과 함께 \'개화당의 급진이 불가능한 일\'이라는 입장을 견지하고 개화당의 급진성을 겨냥, 근신을 촉구하는 입장을 보였다. 며칠 뒤 윤치호는 김옥균에게 "가친(아버지)이 기회를 보고, 변화를 엿보아 움직이는 것이 좋겠다고 한다 "라는 말을 전했다.\\n\\n그는 서광범, 김옥균, 서재필, 박영효 등과 가까이 지냈고 혁명의 성공을 내심 기대하였지만 1884년 12월 갑신정변 때는 개량적 근대화론자로서, 주도층과의 시국관 차이로 적극 참여하지는 않았다. 1884년 12월 4일(음력 10월 17일)에 갑신정변이 발생하자 음력 10월 18일 윤치호와 윤웅렬은 "(개화당)이 무식하여 이치를 모르고, 무지하여 시세에 어두운 것"이라고 논했다 우선 윤치호는 이들의 거사 준비가 허술하고, 거사 기간이 짧다는 점과 인력을 많이 동원하지 못한 점을 보고 실패를 예감하였다. 또한 윤치호는 독립과 개화를 달성하는데 고종 만을 믿을 수는 없다고 봤다.\\n\\n그러나 김옥균, 박영효 등과 절친했기 때문에 정변 실패 후 신변의 위 을 느껴 출국을 결심하게 된다. 사실 갑신정변의 실패를 예감했던 그는 망명할 계획을 미리 세워놓기도 했다.'

question = grouped['question'][0]
# ['윤치호와 함께 개화당의 거사에 대해 비판했던 사람은?', '윤치호가 국방군 양성을 위해 만들자고 건의한 것은?']

tokenizer =  GPT2TokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')



model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
text = f"{context}을 기반으로 질문을 생성하세요:\n\n:"
input_ids = tokenizer.encode(text, return_tensors='pt')
gen_ids = model.generate(input_ids,
                           max_length=600,
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True)
generated = tokenizer.decode(gen_ids[0])

print(generated)


(myenv) root@instance-12466:~/sungeun/level2-mrc-nlp-13# python3 llm_code.py
1884년 1월 18일부터 8월 9일까지 윤치호는 거듭하여 사관학교 설립을 상주한다. 윤치호는 군대 통솔권의 일원화 군인정신의 합일, 상무정신의 강화를 통하여 충성스럽고 용감한 국방군을 양성해야 한다고 보았던 것이다. 이러한 목적에서 그는 미국인 군사교관을 초빙하여 각 영을 통합훈련할 것 과 사관학교 설립을 건의했던 것이다. 이어 병원과 학교의 설립 및 전신국의 설치를 미국인에게 허가해줄 것을 건의하는 등 근대시설의 도입에도 깊은 관심을 보였다\n\n1884년 7월에는 선교사들을 통해 신식 병원과 전화국을 유치, 개설할 것을 고종에게 상주하여 허락받았다. 그러나 신식 병원 도입과 전화국 개통은 갑신정변의 실패로 전면 백지화된다. 1894년 9월 무렵 그는 일본의 조선 침략을 예상하였다. '일본은 이제까지는 개혁을 조선인 스스로 하도록 하려 했다. 그러나 그들이 볼 때 조선인들이 개혁의 의욕도 능력도 없음을 보고 주도권을 잡기로 결심한 것 같다 '며 일본이 한국에 영향력을 행사하리라고 전망했다. 1884년 12월의 갑신정변 직전까지 그는 온건파 개화당의 일원으로 자주독립과 참정권, 부국강병을 위해 활동하였다. 영어 실력의 부족함을 느낀 그는 다시 주조선미국 공사관의 직원들과 교류하며 자신에게 영어를 가르쳐줄 것을 부탁하여 주조선미국공사관 직원 미군 중위 존 B. 베르나든(John B. Bernadon)이 이를 수락하였다. 5월 그는 1개월간 베르나든에게 하루 한 시간씩 영어 개인 지도를 받기도 했다.\n\n1884년 12월 갑신정변 초기에 윤치호는 정변 계획을 접하고 혁명의 성공을 기대하였다. 당시 김옥균을 믿고 따랐던 그는 1894년 11월에 접어들면서 윤치호는 아버지인 윤웅렬과 함께 '개화당의 급진이 불가능한 일'이라는 입장을 견지하고 개화당의 급진성을 겨냥, 근신을 촉구하는 입장을 보였다. 며칠 뒤 윤치호는 김옥균에게 "가친(아버지)이 기회를 보고, 변화를 엿보아 움직이는 것이 좋겠다고 한다 "라는 말을 전했다.\n\n그는 서광범, 김옥균, 서재필, 박영효 등과 가까이 지냈고 혁명의 성공을 내심 기대하였지만 1884년 12월 갑신정변 때는 개량적 근대화론자로서, 주도층과의 시국관 차이로 적극 참여하지는 않았다. 1884년 12월 4일(음력 10월 17일)에 갑신정변이 발생하자 음력 10월 18일 윤치호와 윤웅렬은 "(개화당)이 무식하여 이치를 모르고, 무지하여 시세에 어두운 것"이라고 논했다 우선 윤치호는 이들의 거사 준비가 허술하고, 거사 기간이 짧다는 점과 인력을 많이 동원하지 못한 점을 보고 실패를 예감하였다. 또한 윤치호는 독립과 개화를 달성하는데 고종 만을 믿을 수는 없다고 봤다.\n\n그러나 김옥균, 박영효 등과 절친했기 때문에 정변 실패 후 신변의 위 을 느껴 출국을 결심하게 된다. 사실 갑신정변의 실패를 예감했던 그는 망명할 계획을 미리 세워놓기도 했다.을 기반으로 질문을 생성하세요:

: , . (서울 : 한국학중앙연구원), pp. 330-331.
'''

########################################################################################################
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from datasets import Dataset


# # load file
# base_train = Dataset.load_from_disk('/data/ephemeral/home/sungeun/level2-mrc-nlp-13/data/train_dataset/train')
# base_train_df = base_train.to_pandas()

# # 'title'이 '윤치호'인 행을 먼저 필터링한 후 중복된 'context' 제거
# yunchiho_unique_context_rows = base_train_df[base_train_df['title'] == '윤치호']

# grouped = yunchiho_unique_context_rows.groupby('context')['question'].apply(list).reset_index()
# context = grouped['context'][0]
# # '1884년 1월 18일부터 8월 9일까지 윤치호는 거듭하여 사관학교 설립을 상주한다. 윤치호는 군대 통솔권의 일원화 군인정신의 합일, 상무정신의 강화를 통하여 충성스럽고 용감한 국방군을 양성해야 한다고 보았던 것이다. 이러한 목적에서 그는 미국인 군사교관을 초빙하여 각 영을 통합훈련할 것 과 사관학교 설립을 건의했던 것이다. 이어 병원과 학교의 설립 및 전신국의 설치를 미국인에게 허가해줄 것을 건의하는 등 근대시설의 도입에도 깊은 관심을 보였다\\n\\n1884년 7월에는 선교사들을 통해 신식 병원과 전화국을 유치, 개설할 것을 고종에게 상주하여 허락받았다. 그러나 신식 병원 도입과 전화국 개통은 갑신정변의 실패로 전면 백지화된다. 1894년 9월 무렵 그는 일본의 조선 침략을 예상하였다. \'일본은 이제까지는 개혁을 조선인 스스로 하도록 하려 했다. 그러나 그들이 볼 때 조선인들이 개혁의 의욕도 능력도 없음을 보고 주도권을 잡기로 결심한 것 같다 \'며 일본이 한국에 영향력을 행사하리라고 전망했다. 1884년 12월의 갑신정변 직전까지 그는 온건파 개화당의 일원으로 자주독립과 참정권, 부국강병을 위해 활동하였다. 영어 실력의 부족함을 느낀 그는 다시 주조선미국 공사관의 직원들과 교류하며 자신에게 영어를 가르쳐줄 것을 부탁하여 주조선미국공사관 직원 미군 중위 존 B. 베르나든(John B. Bernadon)이 이를 수락하였다. 5월 그는 1개월간 베르나든에게 하루 한 시간씩 영어 개인 지도를 받기도 했다.\\n\\n1884년 12월 갑신정변 초기에 윤치호는 정변 계획을 접하고 혁명의 성공을 기대하였다. 당시 김옥균을 믿고 따랐던 그는 1894년 11월에 접어들면서 윤치호는 아버지인 윤웅렬과 함께 \'개화당의 급진이 불가능한 일\'이라는 입장을 견지하고 개화당의 급진성을 겨냥, 근신을 촉구하는 입장을 보였다. 며칠 뒤 윤치호는 김옥균에게 "가친(아버지)이 기회를 보고, 변화를 엿보아 움직이는 것이 좋겠다고 한다 "라는 말을 전했다.\\n\\n그는 서광범, 김옥균, 서재필, 박영효 등과 가까이 지냈고 혁명의 성공을 내심 기대하였지만 1884년 12월 갑신정변 때는 개량적 근대화론자로서, 주도층과의 시국관 차이로 적극 참여하지는 않았다. 1884년 12월 4일(음력 10월 17일)에 갑신정변이 발생하자 음력 10월 18일 윤치호와 윤웅렬은 "(개화당)이 무식하여 이치를 모르고, 무지하여 시세에 어두운 것"이라고 논했다 우선 윤치호는 이들의 거사 준비가 허술하고, 거사 기간이 짧다는 점과 인력을 많이 동원하지 못한 점을 보고 실패를 예감하였다. 또한 윤치호는 독립과 개화를 달성하는데 고종 만을 믿을 수는 없다고 봤다.\\n\\n그러나 김옥균, 박영효 등과 절친했기 때문에 정변 실패 후 신변의 위 을 느껴 출국을 결심하게 된다. 사실 갑신정변의 실패를 예감했던 그는 망명할 계획을 미리 세워놓기도 했다.'

# question = grouped['question'][0]

#########################################################################################################
''' Meta-Llama 용량이 너무 커서 저장공간에 피해를 줌...
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

def generate_response(system_message, user_message):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    response = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(response, skip_special_tokens=True)



context = context
questions = question

user_message = f"Context: {context}\nExisting Questions: {questions}"

response = generate_response(system_message="", user_message=user_message)

print(response)
'''

# 모델 준비
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="auto",
)
model.eval()

# 추론
PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
instruction = '''
다음 텍스트를 기반으로 "질문"과 "답"을 생성해줘, 단, 답이 단답형이 되도록 질문을 해야돼:
'1884년 1월 18일부터 8월 9일까지 윤치호는 거듭하여 사관학교 설립을 상주한다. 윤치호는 군대 통솔권의 일원화 군인정신의 합일, 상무정신의 강화를 통하여 충성스럽고 용감한 국방군을 양성해야 한다고 보았던 것이다. 이러한 목적에서 그는 미국인 군사교관을 초빙하여 각 영을 통합훈련할 것 과 사관학교 설립을 건의했던 것이다. 이어 병원과 학교의 설립 및 전신국의 설치를 미국인에게 허가해줄 것을 건의하는 등 근대시설의 도입에도 깊은 관심을 보였다\\n\\n1884년 7월에는 선교사들을 통해 신식 병원과 전화국을 유치, 개설할 것을 고종에게 상주하여 허락받았다. 그러나 신식 병원 도입과 전화국 개통은 갑신정변의 실패로 전면 백지화된다. 1894년 9월 무렵 그는 일본의 조선 침략을 예상하였다. \'일본은 이제까지는 개혁을 조선인 스스로 하도록 하려 했다. 그러나 그들이 볼 때 조선인들이 개혁의 의욕도 능력도 없음을 보고 주도권을 잡기로 결심한 것 같다 \'며 일본이 한국에 영향력을 행사하리라고 전망했다. 1884년 12월의 갑신정변 직전까지 그는 온건파 개화당의 일원으로 자주독립과 참정권, 부국강병을 위해 활동하였다. 영어 실력의 부족함을 느낀 그는 다시 주조선미국 공사관의 직원들과 교류하며 자신에게 영어를 가르쳐줄 것을 부탁하여 주조선미국공사관 직원 미군 중위 존 B. 베르나든(John B. Bernadon)이 이를 수락하였다. 5월 그는 1개월간 베르나든에게 하루 한 시간씩 영어 개인 지도를 받기도 했다.\\n\\n1884년 12월 갑신정변 초기에 윤치호는 정변 계획을 접하고 혁명의 성공을 기대하였다. 당시 김옥균을 믿고 따랐던 그는 1894년 11월에 접어들면서 윤치호는 아버지인 윤웅렬과 함께 \'개화당의 급진이 불가능한 일\'이라는 입장을 견지하고 개화당의 급진성을 겨냥, 근신을 촉구하는 입장을 보였다. 며칠 뒤 윤치호는 김옥균에게 "가친(아버지)이 기회를 보고, 변화를 엿보아 움직이는 것이 좋겠다고 한다 "라는 말을 전했다.\\n\\n그는 서광범, 김옥균, 서재필, 박영효 등과 가까이 지냈고 혁명의 성공을 내심 기대하였지만 1884년 12월 갑신정변 때는 개량적 근대화론자로서, 주도층과의 시국관 차이로 적극 참여하지는 않았다. 1884년 12월 4일(음력 10월 17일)에 갑신정변이 발생하자 음력 10월 18일 윤치호와 윤웅렬은 "(개화당)이 무식하여 이치를 모르고, 무지하여 시세에 어두운 것"이라고 논했다 우선 윤치호는 이들의 거사 준비가 허술하고, 거사 기간이 짧다는 점과 인력을 많이 동원하지 못한 점을 보고 실패를 예감하였다. 또한 윤치호는 독립과 개화를 달성하는데 고종 만을 믿을 수는 없다고 봤다.\\n\\n그러나 김옥균, 박영효 등과 절친했기 때문에 정변 실패 후 신변의 위 을 느껴 출국을 결심하게 된다. 사실 갑신정변의 실패를 예감했던 그는 망명할 계획을 미리 세워놓기도 했다.'
'''

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty = 1.1
)

print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))