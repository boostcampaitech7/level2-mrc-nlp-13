from dataclasses import dataclass, field
from typing import Optional

<<<<<<< HEAD
=======
# init 쉽게 하는것
>>>>>>> 73194be4eea4b3945d1021bca4e23d5fb144c5f7

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
<<<<<<< HEAD
        default="klue/bert-base",
=======
        default="klue/bert-base", # 디폴트 모델은 klue bert base
>>>>>>> 73194be4eea4b3945d1021bca4e23d5fb144c5f7
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
<<<<<<< HEAD
        default=None,
=======
        default=None, # none -> model_name_or_path 따라감 // inference.py 75번째 줄 , 80번째 줄
>>>>>>> 73194be4eea4b3945d1021bca4e23d5fb144c5f7
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
<<<<<<< HEAD
        default=None,
=======
        default=None, # none -> model_name_or_path 따라감 // inference.py 75번째 줄 , 80번째 줄
>>>>>>> 73194be4eea4b3945d1021bca4e23d5fb144c5f7
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
<<<<<<< HEAD
        default="data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
=======
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    # 기존 캐시를 무시하고 태스크를 새로 실행, 최신결과를 얻을 수 있지만 실행 시간이 길어질 수 있다 
>>>>>>> 73194be4eea4b3945d1021bca4e23d5fb144c5f7
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
<<<<<<< HEAD
=======
    # 전처리에 사용가능한 ~ 속도 조절
>>>>>>> 73194be4eea4b3945d1021bca4e23d5fb144c5f7
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
<<<<<<< HEAD
=======
    # 384 를 기준으로 나머지 뒤의 길이들을 padding 할지 안할지에 대해서
>>>>>>> 73194be4eea4b3945d1021bca4e23d5fb144c5f7
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
<<<<<<< HEAD
=======
    # 만약 384 를 기준으로 두개의 context 로 나누면, 겹치는 토큰수
>>>>>>> 73194be4eea4b3945d1021bca4e23d5fb144c5f7
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
<<<<<<< HEAD
=======
    # 텍스트 조각 (passage retrieval) 를 sparse embedding 을 사용할지
>>>>>>> 73194be4eea4b3945d1021bca4e23d5fb144c5f7
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
<<<<<<< HEAD
=======
    # faiss -> 대규모 데이터셋에서 효율적인 유사도 검색을 수행하기 위한 라이브러리
    # FAISS는 데이터를 여러 클러스터로 나누어 검색 공간을 줄인다
    # 이를 위한 cluster 의 수를 설정하는 것
>>>>>>> 73194be4eea4b3945d1021bca4e23d5fb144c5f7
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
