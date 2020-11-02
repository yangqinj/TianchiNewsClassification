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
TEST_PATH=${DATA_DIR}/test.csv


export PATHON_PATH=${SRC_DIR}
cd ${SRC_DIR}

python3 test.py \
    --test_path ${TEST_PATH} \
    --embedding_path ${EMBEDDING_FILE} \
    --model ${MODEL} \
    --model_dir ${MODEL_DIR} \
    --config_dir ${CONFIG_DIR} \
    --log_dir ${LOG_DIR}
