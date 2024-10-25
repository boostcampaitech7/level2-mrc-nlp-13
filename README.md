# Level 2 Project :: ODQA(Open-Domain Question Answering) 

### 📝 Abstract
- 이 프로젝트는 네이버 부스트 캠프 AI-Tech 7기 NLP Level 2 기초 프로젝트 경진대회로, Dacon과 Kaggle과 유사한 대회형 방식으로 진행되었다.
- ODQA(Open-Domain Question Answering) task는 주어진 질문에 대해 대규모 문서 집합에서 관련 정보를 검색하고, 그 정보로부터 정확한 답변을 추출하는 것이 주제로, 모든 팀원이 데이터 전처리부터 앙상블까지 AI 모델링의 전 과정을 함께 협업했다.

<br>

## Project Leader Board 
- Public Leader Board
<img width="450" alt="public_leader_board" src="https://github.com/user-attachments/assets/4d4592bc-1e3e-4455-8de9-cf61e5fc6d50">

- Private Leader Board 
<img width="450" alt="private_leader_board" src="https://github.com/user-attachments/assets/f1a5b53d-f30b-4d87-8a14-3cc1a602f8a0">

- [📈 NLP 13조 Project Wrap-Up report 살펴보기](https://github.com/user-attachments/files/17182231/NLP_13.Wrap-Up.pdf
)

<br>

## 🧑🏻‍💻 Team Introduction & Members 

> Team name : 스빈라킨스배 [ NLP 13조 ]

### 👨🏼‍💻 Members
권지수|김성은|김태원|이한서|정주현|
:-:|:-:|:-:|:-:|:-:
<img src='https://github.com/user-attachments/assets/ab4b7189-ec53-41be-8569-f40619b596ce' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/49dc0e59-93ee-4e08-9126-4a3deca9d530' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/a15b0f0b-cd89-412b-9b3d-f59eb9787613' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/11b2ed88-bf94-4741-9df5-5eb2b9641a9b' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/3e2d2a7e-1c64-4cb7-97f6-a2865de0c594' height=125 width=100></img>
[Github](https://github.com/Kwon-Jisu)|[Github](https://github.com/ssungni)|[Github](https://github.com/chris40461)|[Github](https://github.com/beaver-zip)|[Github](https://github.com/peter520416)
<a href="mailto:wltn80609@ajou.ac.kr" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:sunny020111@ajou.ac.kr" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:chris40461@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:beaver.zip@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:peter520416@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|

### 🧑🏻‍🔧 Members' Role

| 이름 | 역할 |
| :---: | --- |
| **`권지수`** | **EDA** (라벨 분포 데이터분석), **모델 탐색** (KLUE: 논문 바탕으로 RoBERTa와 ELECTRA 계열 모델 중심으로 탐색), **모델 실험** (team-lucid/deberta-v3-base-korean), **Ensemble 실험** (output 평균 및 가중치 활용) |
| **`김성은`** | **EDA** (라벨 분포 데이터분석), **모델 탐색** (Encoder, Decoder, Encoder - Decoder 모델로 세분화하여 탐색), **모델 실험** (snunlp-KR-ELECTRA), **Ensemble 실험** (output 평균 및 가중치 활용) |
| **`김태원`** | **모델 실험** (KR-ELECTRA-discriminator, electra-kor-base, deberta-v3, klue-roberta ), **데이터 증강** (label rescaling(0점 인덱스의 제거 및 5점 인덱스 추가), 단순 복제 데이터 증강(1점~3점 인덱스), train 데이터의 전체적인 맞춤법 교정/불용어 제거/띄어쓰기 교정), **모델 Ensemble** (weighted sum for 3model/4models) |
| **`이한서`** |**데이터 증강**(조사 대체, Label 분포 균형화), **모델 실험**(team-lucid/deberta-v3-base-korean, monologg/koelectra-base-v3-discriminator, snunlp/KR-ELECTRA), Hyperparameter Tuning(Optuna Template 제작 및 실험)|
| **`정주현`** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | **데이터 EDA** (Label 분포, 문장 내의 단어 빈도), **데이터 증강** (Swap sentence1 and sentence2, 유의어 교체(‘너무’, ‘진짜’, ‘정말’)), **모델 선정 및 Ensemble** (T5-base-korean-summarization), Ensemble(Blending Ensemble for 3 or 4 model(meta model = Ridge)) |

<br>

## 🖥️ Project Introduction 


|**프로젝트 주제**| **Open-Domain Question Answering: ** 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 NLP Task|
| :---: | --- |
|**프로젝트 구현내용**| 1. Hugging Face 의 Pretrained 모델과 STS 데이터셋을 활용해 두 문장의 0 과 5 사이의 유사도를 측정하는 AI 모델을 구축 <br>2. 리더보드 평가지표인 피어슨 상관 계수(Pearson Correlation Coefficient ,PCC)에서 높은 점수(1에 가까운 점수)에 도달할 수 있도록 데이터 전처리, 증강, 하이퍼 파라미터 튜닝을 진행|
|**개발 환경**|**• `GPU` :** Tesla V100 서버 4개 (RAM32G)<br> **• `개발 Tool` :** Jupyter notebook, VS Code [서버 SSH연결]
|**협업 환경**|**• `Github Repository` :** Baseline 코드 공유 및 버전 관리, 개인 branch를 사용해 작업상황 공유 <br>**• `Notion` :** STS 프로젝트 페이지를 통한 역할분담, 실험 가설 설정 및 결과 공유 <br>**• `SLACK, Zoom` :** 실시간 대면/비대면 회의|

<br>

## 📁 Project Structure

### 🗂️ 디렉토리 구조 설명
- 학습 데이터 경로: `./data`
- 학습 메인 코드: `./train.py`
- 학습 데이터셋 경로: `./data/train_dataset/train`
- 테스트 데이터셋 경로: `./data/train_dataset/validation`

### 📄 코드 구조 설명

> script 파일을 생성하여, 하이퍼 파라미터의 조정 및 train,test,ensemble 을 용이하게 했다.

- **Dense Retriever Train** : `dense_train.py`
- **Train** : `train.sh`
- **Predict** : `test.sh`
- **Ensemble** : `softvoting.py`
- **최종 제출 파일** : `./ensemble/predictions.json`

```
📦 base
┣ 📂 dense_model
┣ 📂 ensemble
┃ ┗ diff.py
┣ 📂 models
┣ 📂 nbest
┣ arguments.py
┣ dense_encoder.py
┣ dense_train.py
┣ dense_util.py
┣ eval.sh
┣ inference.py
┣ requirements.txt
┣ retrieval.py
┣ softvoting.py
┣ test.sh
┣ train.py
┣ train.sh
┣ trainer_qa.py
┗ utils_qa.py
 ```
<br>

## 📐 Project Ground Rule
>팀 협업을 위해 프로젝트 관련 Ground Rule을 설정하여 프로젝트가 원활하게 돌아갈 수 있도록 규칙을 정했으며, 날짜 단위로 간략한 목표를 설정하여 협업을 원활하게 진행할 수 있도록 계획하여 진행했다.

**- a. `Server 관련`** : 권지수, 김성은, 이한서, 정주현 캠퍼는 각자 서버를 생성해 모델 실험을 진행하고, 김태원 캠퍼는 서버가 유휴 상태인 서버에서 실험을 진행했다.

**- b. `Git 관련`** : 각자 branch 생성해 작업하고, 공통으로 사용할 파일은 main에 push 하는 방법으로 협업했다.

**- c. `Submission 관련`** : 대회 마감 2일 전까지는 자유롭게 제출했고, 2일 전부터는 인당 2회씩 분배했다.

**- d. `Notion 관련`** : 원활한 아이디어 브레인스토밍과 분업을 위해 회의를 할 경우 노션에 기록하며, 연구 및 실험결과의 기록을 공유했다.

<br>

## 🗓 Project Procedure: 총 14일 진행
- **`(1~3 일차)`**: EDA 분석
- **`(3~5 일차)`**: 데이터 전처리
- **`(6~11 일차)`** : 데이터 증강
- **`(7~12 일차)`** : 모델링 및 튜닝
- **`(11~13 일차)`** : 앙상블

* 아래는 저희 프로젝트 진행과정을 담은 Gantt차트 입니다. 
<img width="800" alt="Gantt" src="https://github.com/user-attachments/assets/23c3dacb-95c7-49c0-88df-ebc5b5f9b1a7">

<br>

## **MRC**
* 우리는 먼저 Retriever - Reader 모델을 구현하기에 앞서, KorQuad data 에 대해서 pre-trained 된 모델을 사용해, 부족한 Train dataset 을 보강하여 학습하기로 하였다.
<br>

## **Retriever**
* Retriever 모델의 경우, KorQuad data 를 통해 question, passage embedding 을 미리 학습하는 과정을 가진 Dense Retriever 과, BM25 를 사용한 Sparse Retriever 을 결합한 Hybrid Retriever 을 사용했다.
* Hybrid Retriever 을 하는 방식은 크게 3가지 인데, 이 3가지를 모두 활용하여 passage 와 question 의 연관성을 최대로 학습하고, 사용하고자 하였다.

| **Score Function Type**                       | **Description**                                                                                        |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **1. λ x Dense Similarity + BM25 Score** | Dense Retriever와 BM25 점수를 결합하여 질문과 문서 간의 관계를 강화합니다. λ는 두 점수 간의 가중치를 조정합니다. |
| **2. Reciprocal Rank Fusion (RRF)**      | 여러 Retriever의 랭킹을 기반으로 각 문서의 reciprocal rank를 합산하여 최종 점수를 계산합니다. |
| **3. (λ x Dense Similarity + BM25 Score) → ko-reranker** | 1번의 Score 를 한국어 re-ranker에 입력하여 최종 랭킹을 정제하고 성능을 향상시킵니다.     |
<br>

## **Ensemble Model**

* 최종적으로 3개의 json 파일을 softvoting 기법을 활용하여 사용했했다.

| **File Name**                             | **Score Function**                                   |
|-------------------------------------------|-----------------------------------------------------|
| nbest_predictions_op1.json               | λ x Dense Similarity + BM25 Score                   |
| nbest_predictions_op2.json               | Reciprocal Rank Fusion (RRF)                        |
| nbest_predictions_op3.json               | (λ x Dense Similarity + BM25 Score) → ko-reranker   |

<br>

## 💻 Getting Started

### ⚠️  How To install Requirements
```bash
# 필요 라이브러리 설치
pip install -r requirements.txt
```

### ⌨️ How To Train & Test
```bash
#데이터 증강
python3 augmentation.py
# train.py 코드 실행 : 모델 학습을 순차적으로 진행
# 이후 test.py 코드를 순차적으로 실행하여 test
# config.yaml 내 모델 이름, lr 을 리스트 순서대로 변경하며 train 으로 학습

#plm_name[0], lr[0] -> klue/roberta-base
python3 train.py
python3 test.py

#plm_name[1], lr[1] -> kykim/electra-kor-base
python3 train.py
python3 test.py

#plm_name[2], lr[2] -> team-lucid/deberta-v3-base-korean 
python3 train.py
python3 test.py

#plm_name[3], lr[3] -> snunlp/KR-ELECTRA-discriminator
python3 train.py
python3 test.py

#plm_name[4], lr[4] -> eenzeenee/t5-base-korean- summarization
python3 train.py
python3 test.py

```

### ⌨️ How To Ensemble
```bash
# 순차적으로 weighted ensemble 진행 후, 출력 결과를 사용해서 blended ensemble 진행
python3 weighted_ensemble.py # klue/roberta-base, eenzeenee/t5-base-korean-summarization, kykim/electra-kor-base
python3 blending_ensemble.py # kykim/electra-kor-base , team-lucid/deberta-v3-base-korean , snunlp/KR-ELECTRA-discriminator, weighted_ensemble
```
