#!/bin/bash

# 수정 가능한 매개변수
MODEL_PATH="./models/train_dataset"
OUTPUT_DIR="./nbest"
DATA_DIR="../data/test_dataset"
MAX_SEQ_LENGTH=384
DOC_STRIDE=128
MAX_ANSWER_LENGTH=50
BATCH_SIZE=32
EVAL_RETRIEVAL=true
NUM_CLUSTERS=64
TOP_K_RETRIEVAL=10
USE_FAISS=false
RETRIEVER_TYPE="hybrid"

python inference.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATA_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --doc_stride $DOC_STRIDE \
    --max_answer_length $MAX_ANSWER_LENGTH \
    --per_device_eval_batch_size $BATCH_SIZE \
    --eval_retrieval $EVAL_RETRIEVAL \
    --num_clusters $NUM_CLUSTERS \
    --top_k_retrieval $TOP_K_RETRIEVAL \
    --use_faiss $USE_FAISS \
    --overwrite_output_dir \
    --retriever_type $RETRIEVER_TYPE\
    --hybrid_score_option 1\
    --prefix "op1"\
    --do_predict

python inference.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATA_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --doc_stride $DOC_STRIDE \
    --max_answer_length $MAX_ANSWER_LENGTH \
    --per_device_eval_batch_size $BATCH_SIZE \
    --eval_retrieval $EVAL_RETRIEVAL \
    --num_clusters $NUM_CLUSTERS \
    --top_k_retrieval $TOP_K_RETRIEVAL \
    --use_faiss $USE_FAISS \
    --overwrite_output_dir \
    --retriever_type $RETRIEVER_TYPE\
    --hybrid_score_option 2\
    --prefix "op2"\
    --do_predict

python inference.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATA_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --doc_stride $DOC_STRIDE \
    --max_answer_length $MAX_ANSWER_LENGTH \
    --per_device_eval_batch_size $BATCH_SIZE \
    --eval_retrieval $EVAL_RETRIEVAL \
    --num_clusters $NUM_CLUSTERS \
    --top_k_retrieval $TOP_K_RETRIEVAL \
    --use_faiss $USE_FAISS \
    --overwrite_output_dir \
    --retriever_type $RETRIEVER_TYPE\
    --hybrid_score_option 3\
    --prefix "op3"\
    --do_predict