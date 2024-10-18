import json
import os
import pickle
import time
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from dense_encoder import RobertaEncoder
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt 
from collections import defaultdict

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

    def get_sparse_embedding(self) -> NoReturn:
        print("BM25Okapi 모델 사용 중, 별도의 임베딩이 필요하지 않습니다.")
        self.embedding_ready = True  # 임베딩이 준비되었다는 플래그 설정


    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.embedding_ready, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, topk)

            print("[Search query]\n", query_or_dataset, "\n")
            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[i] for i in doc_indices])

        elif isinstance(query_or_dataset, Dataset):
            doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset, topk)
            
            # 결과를 데이터프레임으로 변환
            total = []
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: Building dataset")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)
            
            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, topk: int) -> Tuple[List[float], List[int]]:
        tokenized_query = self.tokenize_fn(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        doc_indices = np.argsort(doc_scores)[::-1][:topk]
        return doc_scores[doc_indices].tolist(), doc_indices.tolist()

    def get_relevant_doc_bulk(self, dataset: Dataset, topk: int) -> Tuple[List[List[float]], List[List[int]]]:
        doc_scores = []
        doc_indices = []
        for example in tqdm(dataset, desc="Sparse retrieval:"):
            scores, indices = self.get_relevant_doc(example["question"], topk)
            doc_scores.append(scores)
            doc_indices.append(indices)
        
        return doc_scores, doc_indices
        
class DenseRetrieval:
    def __init__(
            self,
            args,
            p_encoder_path,
            q_encoder_path,
            data_path: Optional[str] = "../data/",
            context_path: Optional[str] = "wikipedia_documents.json",
            ):

        self.data_path = data_path
        self.config = AutoConfig.from_pretrained(args.config_name_dpr)
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        
        self.p_encoder = RobertaEncoder(self.config)
        self.p_encoder.load_state_dict(torch.load(p_encoder_path))
        
        self.q_encoder = RobertaEncoder(self.config)
        self.q_encoder.load_state_dict(torch.load(q_encoder_path))
        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()
        # self.tokenizer = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.config_name_dpr)

    def get_dense_embedding(self):
        pickle_name = f"dense_embedding.npy"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            self.p_embedding = torch.load(emd_path)
            self.p_embedding = self.p_embedding.to('cuda' if torch.cuda.is_available() else 'cpu')
            print("Embedding pickle loaded.")
        else:
            print("Building passage embedding")
            self.p_embedding = self.passage_embedding(self.contexts)
            print(self.p_embedding.shape)
            torch.save(self.p_embedding, emd_path)
            print("Embedding pickle saved.")
            
    def passage_embedding(self, valid_corpus):
        p_embs = []
        with torch.no_grad():
            self.p_encoder.eval()
            for p in tqdm(valid_corpus, total=len(valid_corpus), desc="Dense Embedding Create..."):
                inputs = self.tokenizer(p, padding="max_length", truncation=True, return_tensors='pt')
                inputs = {key: value.to('cuda' if torch.cuda.is_available() else 'cpu') for key, value in inputs.items()}
                p_emb = self.p_encoder(**inputs).cpu().numpy()
                p_embs.append(p_emb)
        torch.cuda.empty_cache()
        p_embs = torch.Tensor(p_embs).squeeze()
        return p_embs
    
    def query_embedding(self, queries):
        with torch.no_grad():
            self.q_encoder.eval()
            if isinstance(queries, str):
                queries = [queries]
            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            q_emb = self.q_encoder(**q_seqs_val) 
        return q_emb

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            total = []
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        query_vec = self.query_embedding([query])
        assert query_vec.sum().item() != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = torch.matmul(query_vec, self.p_embedding.T)
        
        sorted_result = torch.argsort(result.squeeze(), descending=True)
        doc_score = result.squeeze()[sorted_result][:k].tolist()
        doc_indices = sorted_result[:k].tolist()
        return doc_score, doc_indices
        
    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        query_vec = self.query_embedding(queries)
        device = query_vec.device
        
        with torch.no_grad():
            result = torch.matmul(query_vec, self.p_embedding.T)
            doc_scores = []
            doc_indices = []
            for i in range(result.shape[0]):
                sorted_result = torch.argsort(result[i, :], descending=True)
                doc_scores.append(result[i, :][sorted_result][:k].tolist())
                doc_indices.append(sorted_result[:k].tolist())
        return doc_scores, doc_indices
    
class HybridRetrieval:
    def __init__(
            self,
            args,
            tokenize_fn,
            p_encoder_path,
            q_encoder_path,
            data_path: Optional[str] = "../data/",
            context_path: Optional[str] = "wikipedia_documents.json",
            option : int = 0, # get hybrid scores option -> score function 바꾸려면 이거 바꾸셈
            lambda_weight: float = 0.5
    ):
        self.sparse_retrieval = SparseRetrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path
        )
        self.dense_retrieval = DenseRetrieval(
            args=args, 
            p_encoder_path=p_encoder_path, 
            q_encoder_path=q_encoder_path,
            data_path=data_path,
            context_path=context_path
        )
        self.initialize_reranker()
        self.q_encoder = self.dense_retrieval.q_encoder
        self.lambda_weight = lambda_weight
        self.option = option

        print('Sparse Embedding Get Start')
        self.sparse_retrieval.get_sparse_embedding()
        print('Sparse Embedding Get End')

        print('Dense Embedding Get Start')
        self.dense_retrieval.get_dense_embedding()
        print('Dense Embedding Get End')

        self.p_embedding = self.dense_retrieval.p_embedding
        if torch.cuda.is_available():
            self.p_embedding = torch.Tensor(self.p_embedding).to("cuda")

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        
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
    
    def retrieve(self, query_or_dataset, topk=1):
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_hybrid_scores(query_or_dataset, topk)
            return doc_scores, [self.contexts[i] for i in doc_indices]

        elif isinstance(query_or_dataset, Dataset):
            total = []
            for example in tqdm(query_or_dataset, desc="Hybrid retrieval"):
                scores, indices = self.get_hybrid_scores(example["question"], topk)
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[i] for i in indices]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)
    #      *************************get_hybrid_score*****************************
    
    def get_hybrid_scores(self, query: str, topk: int) -> Tuple[List[float], List[int]]:
        if self.option == 0:
        # 단순히 BM25 + rerank --> 현재 Sota 리트리버 모델
            sparse_scores, sparse_indices = self.sparse_retrieval.get_relevant_doc(query, topk*2)
            
            initial_contexts = [self.contexts[i] for i in sparse_indices]
            selected_scores = [sparse_scores[i] for i in range(len(sparse_indices))]
            reranked_scores, reranked_contexts = self.rerank(query, initial_contexts, selected_scores, topk)
            reranked_indices = [self.contexts.index(context) for context in reranked_contexts]
        
            return reranked_scores.tolist(), reranked_indices
        
        elif self.option == 1:
            # sparse, dense 로 우선 linear 결합 -> 이후 rerank 를 통해 layer 를 쌓는 과정을 진행
            sparse_scores, sparse_indices = self.sparse_retrieval.get_relevant_doc(query, topk*2)
            dense_scores, dense_indices = self.dense_retrieval.get_relevant_doc(query, topk*2)

            # 점수를 결합합니다
            combined_scores = {}
            for idx in set(sparse_indices + dense_indices):
                bm25_score = sparse_scores[sparse_indices.index(idx)] if idx in sparse_indices else 0
                sim_score = dense_scores[dense_indices.index(idx)] if idx in dense_indices else 0
                combined_scores[idx] = bm25_score + self.lambda_weight * sim_score

            # 결합된 점수로 정렬합니다
            sorted_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)

            # 재순위화를 위한 문서 수를 결정합니다 (available_docs와 topk*2 중 작은 값)
            available_docs = len(sorted_indices)
            docs_to_rerank = min(available_docs, topk*2)

            # 재순위화를 위한 컨텍스트를 준비합니다
            contexts_to_rerank = [self.contexts[i] for i in sorted_indices[:docs_to_rerank]]
            scores_to_rerank = [combined_scores[i] for i in sorted_indices[:docs_to_rerank]]

            # 재순위화를 수행합니다
            reranked_scores, reranked_contexts = self.rerank(query, contexts_to_rerank, scores_to_rerank, topk)

            # 재순위화된 결과의 인덱스를 찾습니다
            reranked_indices = [self.contexts.index(context) for context in reranked_contexts]

            return reranked_scores.tolist(), reranked_indices
        elif self.option == 2:
            # RRF -> https://abluesnake.tistory.com/180 참고하세요
            
            sparse_scores, sparse_indices = self.sparse_retrieval.get_relevant_doc(query, 100)
            dense_scores, dense_indices = self.dense_retrieval.get_relevant_doc(query, 100)

            # RRF를 위한 점수 계산
            rrf_scores = defaultdict(float)
            k = 60  # RRF 상수, 경험적으로 60이 좋은 성능을 보입니다

            # BM25 결과에 대한 RRF 점수 계산
            for rank, idx in enumerate(sparse_indices, 1):
                rrf_scores[idx] += 1 / (rank + k)

            # DPR 결과에 대한 RRF 점수 계산
            for rank, idx in enumerate(dense_indices, 1):
                rrf_scores[idx] += 1 / (rank + k)

            # RRF 점수로 정렬
            sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:topk]
            sorted_scores = [rrf_scores[idx] for idx in sorted_indices]

            return sorted_scores, sorted_indices
        elif self.option == 3:
            # 단순히 sparse, dense 값을 가중치(lambda) 를 주고 Linear 계산
            sparse_scores, sparse_indices = self.sparse_retrieval.get_relevant_doc(query, topk*2)
            dense_scores, dense_indices = self.dense_retrieval.get_relevant_doc(query, topk*2)
            combined_scores = {}
            
            for idx in set(sparse_indices + dense_indices):
                bm25_score = sparse_scores[sparse_indices.index(idx)] if idx in sparse_indices else 0
                sim_score = dense_scores[dense_indices.index(idx)] if idx in dense_indices else 0
                combined_scores[idx] = bm25_score + self.lambda_weight * sim_score

            sorted_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:topk]
            sorted_scores = [combined_scores[i] for i in sorted_indices]

            return sorted_scores, sorted_indices