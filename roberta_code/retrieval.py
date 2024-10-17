# import json
# import os
# import pickle
# import time
# import random
# from contextlib import contextmanager
# from typing import List, NoReturn, Optional, Tuple, Union
# from konlpy.tag import Okt

# import faiss
# import numpy as np
# import pandas as pd
# from datasets import Dataset, concatenate_datasets, load_from_disk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from rank_bm25 import BM25Okapi # BM25 추가
# from tqdm.auto import tqdm
# from collections import OrderedDict

# from konlpy.tag import Mecab

# seed = 2024
# random.seed(seed) # python random seed 고정
# np.random.seed(seed) # numpy random seed 고정



# @contextmanager
# def timer(name):
#     t0 = time.time()
#     yield
#     print(f"[{name}] done in {time.time() - t0:.3f} s")


# class SparseRetrieval:
#     def __init__(
#         self,
#         tokenize_fn,
#         data_path: Optional[str] = "../data",
#         context_path: Optional[str] = "wikipedia_documents.json",
#     ) -> NoReturn:

#         """
#         Arguments:
#             tokenize_fn:
#                 기본 text를 tokenize해주는 함수입니다.
#                 아래와 같은 함수들을 사용할 수 있습니다.
#                 - lambda x: x.split(' ')
#                 - Huggingface Tokenizer
#                 - konlpy.tag의 Mecab

#             data_path:
#                 데이터가 보관되어 있는 경로입니다.

#             context_path:
#                 Passage들이 묶여있는 파일명입니다.

#             data_path/context_path가 존재해야합니다.

#         Summary:
#             Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
#         """
#         self.tokenize_fn = tokenize_fn
#         self.data_path = data_path
#         full_path = os.path.join(self.data_path, context_path)
#         with open(full_path, "r", encoding="utf-8") as f:
#             wiki = json.load(f)
#         self.contexts = list(
#             dict.fromkeys([v["text"] for v in wiki.values()])
#         ) 
#         '''unique_docs = OrderedDict()
#         for v in wiki.values():
#             text = v["text"]
#             title = v["title"]
            
#             # text를 키로 사용하여 중복 제거
#             if text not in unique_docs:
#                 unique_docs[text] = title
#         # OrderedDict의 키와 값을 리스트로 변환
#         self.contexts = list(unique_docs.keys())
#         self.titles = list(unique_docs.values())'''

#         print(f"Lengths of unique contexts : {len(self.contexts)}")
#         self.ids = list(range(len(self.contexts)))
        
#         # 텍스트 토크나이징
#         self.tokenized_contexts = [tokenize_fn(doc) for doc in self.contexts]
#         # BM25 객체 생성
#         self.bm25 = BM25Okapi(self.tokenized_contexts, k1=2.0, b=1.0)

#         self.p_embedding = None  # get_sparse_embedding()로 생성합니다
#         self.indexer = None  # build_faiss()로 생성합니다.

#     def get_sparse_embedding(self) -> NoReturn:
#         """
#         BM25에서는 별도의 Embedding이 필요 없으므로, 이 함수는 생략 가능하거나 로그를 출력하는 정도로 간단히 처리합니다.
#         """
#         print("BM25Okapi 모델 사용 중, 별도의 임베딩이 필요하지 않습니다.")
#         self.embedding_ready = True  # 임베딩이 준비되었다는 플래그 설정


#     def retrieve(
#         self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
#     ) -> Union[Tuple[List, List], pd.DataFrame]:

#         assert self.embedding_ready, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

#         if isinstance(query_or_dataset, str):
#             # Query를 토크나이징
#             tokenized_query = self.tokenize_fn(query_or_dataset)

#             # BM25로 관련 문서 검색
#             doc_scores = self.bm25.get_scores(tokenized_query)
#             doc_indices = np.argsort(doc_scores)[::-1][:topk]

#             print("[Search query]\n", query_or_dataset, "\n")
#             for i in range(topk):
#                 print(f"Top-{i+1} passage with score {doc_scores[doc_indices[i]]:4f}")
#                 print(self.contexts[doc_indices[i]])

#             return (doc_scores[doc_indices], [self.contexts[i] for i in doc_indices])

#         elif isinstance(query_or_dataset, Dataset):
#             total = []
#             for example in tqdm(query_or_dataset, desc="BM25 retrieval"):
#                 tokenized_query = self.tokenize_fn(example["question"])
#                 doc_scores = self.bm25.get_scores(tokenized_query)
#                 doc_indices = np.argsort(doc_scores)[::-1][:topk]

#                 tmp = {
#                     "question": example["question"],
#                     "id": example["id"],
#                     "context": " ".join([self.contexts[pid] for pid in doc_indices]),
#                 }
#                 if "context" in example.keys() and "answers" in example.keys():
#                     tmp["original_context"] = example["context"]
#                     tmp["answers"] = example["answers"]
#                 total.append(tmp)

#             cqas = pd.DataFrame(total)
#             return cqas

# if __name__ == "__main__":

#     import argparse

#     parser = argparse.ArgumentParser(description="")
#     parser.add_argument(
#         "--dataset_name", metavar="./data/train_dataset", type=str, help=""
#     )
#     parser.add_argument(
#         "--model_name_or_path",
#         metavar="bert-base-multilingual-cased",
#         type=str,
#         help="",
#     )
#     parser.add_argument("--data_path", metavar="./data", type=str, help="")
#     parser.add_argument(
#         "--context_path", metavar="wikipedia_documents", type=str, help=""
#     )
#     parser.add_argument("--use_faiss", metavar=False, type=bool, help="")

#     args = parser.parse_args()

#     # Test sparse
#     org_dataset = load_from_disk(args.dataset_name)
#     full_ds = concatenate_datasets(
#         [
#             org_dataset["train"].flatten_indices(),
#             org_dataset["validation"].flatten_indices(),
#         ]
#     )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
#     print("*" * 40, "query dataset", "*" * 40)
#     print(full_ds)

#     from transformers import AutoTokenizer

#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,)

#     retriever = SparseRetrieval(
#         tokenize_fn=Mecab.morphs,
#         data_path=args.data_path,
#         context_path=args.context_path,
#     )

#     query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

#     if args.use_faiss:

#         # test single query
#         with timer("single query by faiss"):
#             scores, indices = retriever.retrieve_faiss(query)

#         # test bulk
#         with timer("bulk query by exhaustive search"):
#             df = retriever.retrieve_faiss(full_ds)
#             df["correct"] = df["original_context"] == df["context"]

#             print("correct retrieval result by faiss", df["correct"].sum() / len(df))

#     else:
#         with timer("bulk query by exhaustive search"):
#             df = retriever.retrieve(full_ds)
#             df["correct"] = df["original_context"] == df["context"]
#             print(
#                 "correct retrieval result by exhaustive search",
#                 df["correct"].sum() / len(df),
#             )

#         with timer("single query by exhaustive search"):
#             scores, indices = retriever.retrieve(query)


### reranker
import json
import os
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from rank_bm25 import BM25Okapi # BM25 추가
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import normalize

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        self.tokenize_fn = tokenize_fn
        self.initialize_reranker()  # reranker 초기화
        self.data_path = data_path
        full_path = os.path.join(self.data_path, context_path)
        with open(full_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        ) 
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        
        # 텍스트 토크나이징
        self.tokenized_contexts = [tokenize_fn(doc) for doc in self.contexts]
        # BM25 객체 생성
        self.bm25 = BM25Okapi(self.tokenized_contexts)

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.
        
    def initialize_reranker(self):
        self.reranker_tokenizer = AutoTokenizer.from_pretrained("Dongjin-kr/ko-reranker")
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained("Dongjin-kr/ko-reranker")
        self.reranker_model.eval()
        if torch.cuda.is_available():
            self.reranker_model = self.reranker_model.cuda()
            
    def rerank(self, query, contexts, scores, topk):
        inputs = self.reranker_tokenizer(
            [query] * len(contexts),
            contexts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.reranker_model(**inputs)
            rerank_scores = outputs.logits.squeeze(-1).cpu().numpy()
        
        reranked_indices = np.argsort(rerank_scores)[::-1][:topk]
        reranked_contexts = [contexts[i] for i in reranked_indices]
        reranked_scores = rerank_scores[reranked_indices]
        
        return reranked_scores, reranked_contexts
    
    def get_sparse_embedding(self) -> NoReturn:
        print("BM25Okapi 모델 사용 중, 별도의 임베딩이 필요하지 않습니다.")
        self.embedding_ready = True  # 임베딩이 준비되었다는 플래그 설정

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.embedding_ready, "get_sparse_embedding() 메서드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            tokenized_query = self.tokenize_fn(query_or_dataset)
            doc_scores = self.bm25.get_scores(tokenized_query)
            doc_indices = np.argsort(doc_scores)[::-1][:topk*2]  # 더 많은 문서를 검색

            initial_contexts = [self.contexts[i] for i in doc_indices]
            reranked_scores, reranked_contexts = self.rerank(query_or_dataset, initial_contexts, doc_scores[doc_indices], topk)

            print("[Search query]\n", query_or_dataset, "\n")
            for i in range(topk):
                print(f"Top-{i+1} passage with score {reranked_scores[i]:4f}")
                print(reranked_contexts[i])

            return (reranked_scores, reranked_contexts)

        elif isinstance(query_or_dataset, Dataset):
            total = []
            for example in tqdm(query_or_dataset, desc="BM25 retrieval + Reranking"):
                tokenized_query = self.tokenize_fn(example["question"])
                doc_scores = self.bm25.get_scores(tokenized_query)
                doc_indices = np.argsort(doc_scores)[::-1][:topk*2]

                initial_contexts = [self.contexts[i] for i in doc_indices]
                reranked_scores, reranked_contexts = self.rerank(example["question"], initial_contexts, doc_scores[doc_indices], topk)

                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(reranked_contexts),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas