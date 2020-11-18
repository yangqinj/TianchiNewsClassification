#!/bin/bash

CUR_DIR=$(pwd)
PROJECT_DIR=${CUR_DIR}/..
SRC_DIR=${PROJECT_DIR}/src
DATA_DIR=${PROJECT_DIR}/data
EMBEDDING_DIR=${PROJECT_DIR}/embeddings
CONFIG_DIR=${PROJECT_DIR}/config
LOG_DIR=${PROJECT_DIR}/log
MODEL_DIR=${PROJECT_DIR}/model


MODEL="TextCNN"
EMBEDDING_FILE=${EMBEDDING_DIR}/word2vec.wv
TRAIN_PATH=${DATA_DIR}/train.csv
N_SPLITS=1


cd ${SRC_DIR}

python3 train.py \
    --train_path ${TRAIN_PATH} \
    --embedding_path ${EMBEDDING_FILE} \
    --model ${MODEL} \
    --model_dir ${MODEL_DIR} \
    --nsplits ${N_SPLITS} \
    --config_dir ${CONFIG_DIR} \
    --log_dir ${LOG_DIR} > ${CUR_DIR}/log_${MODEL}_train_$(date '+%Y%m%d_%H%M%S') 2>&1 &
