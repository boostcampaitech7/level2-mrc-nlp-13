#!/bin/bash

# 수정 가능한 매개변수
MODEL_NAME="uomnf97/klue-roberta-finetuned-korquad-v2"
OUTPUT_DIR="./models/train_dataset"
DATA_DIR="../data/train_dataset"
MAX_SEQ_LENGTH=384
DOC_STRIDE=128
MAX_ANSWER_LENGTH=50
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=3
LEARNING_RATE=1.5e-5
NUM_EPOCHS=2
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.06

python train.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATA_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --doc_stride $DOC_STRIDE \
    --max_answer_length $MAX_ANSWER_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --overwrite_output_dir \
    --do_train