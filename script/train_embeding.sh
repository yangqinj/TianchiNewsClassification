#!/bin/bash

CUR_DIR=$(pwd)
PROJECT_DIR=${CUR_DIR}/..
SRC_DIR=${PROJECT_DIR}/src
DATA_DIR=${PROJECT_DIR}/data
EMBEDDING_DIR=${PROJECT_DIR}/embeddings


export PATHON_PATH=${SRC_DIR}
cd ${SRC_DIR}

python3 train_embeddings.py \
    --dataset_path ${DATA_DIR}/train_2w.csv \
    --size 300 \
    --embedding_path ${EMBEDDING_DIR}/word2vec.wv