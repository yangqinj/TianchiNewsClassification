#!/bin/bash

CUR_DIR=$(pwd)
PROJECT_DIR=${CUR_DIR}/..
SRC_DIR=${PROJECT_DIR}/src
DATA_DIR=${PROJECT_DIR}/data

TRAIN_PATH=${DATA_DIR}/train_2w.csv
TEST_PATH=${DATA_DIR}/test_5k.csv

PREPROCESSED_TRAIN_PATH=${DATA_DIR}/train.csv
PREPROCESSED_TEST_PATH=${DATA_DIR}/test.csv

# remove stopwords
sed "s/3750//g; s/900//g; s/648//g" ${TRAIN_PATH} > ${PREPROCESSED_TRAIN_PATH}
sed "s/3750//g; s/900//g; s/648//g" ${TEST_PATH} > ${PREPROCESSED_TEST_PATH}


# remove label for train dataset
CORPUS_PATH=${DATA_DIR}/all_data.csv
sed 's/^[0-9]*//g' ${PREPROCESSED_TRAIN_PATH} > ${CORPUS_PATH}
cat ${PREPROCESSED_TEST_PATH} >> ${CORPUS_PATH}


head -n 20001 ${PREPROCESSED_TRAIN_PATH} > ${DATA_DIR}/train_2w.csv
head -n 5001 ${PREPROCESSED_TEST_PATH} > ${DATA_DIR}/test_5k.csv