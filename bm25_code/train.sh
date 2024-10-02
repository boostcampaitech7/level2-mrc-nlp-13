#!/bin/bash

# 수정 가능한 매개변수
MODEL_NAME="klue/bert-base"
OUTPUT_DIR="/data/ephemeral/home/jisu/level2-mrc-nlp-13/models/train_datasets"
DATA_DIR="/data/ephemeral/home/data/train_dataset"
MAX_SEQ_LENGTH=384
DOC_STRIDE=128
MAX_ANSWER_LENGTH=30
BATCH_SIZE=16
LEARNING_RATE=5e-5
NUM_EPOCHS=10

python train.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATA_DIR \
    --do_train \
    --max_seq_length $MAX_SEQ_LENGTH \
    --doc_stride $DOC_STRIDE \
    --max_answer_length $MAX_ANSWER_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --overwrite_output_dir