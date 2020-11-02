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
sed 's/^[0-9]*//g' ${PREPROCESSED_TRAIN_PATH} > ${PREPROCESSED_TRAIN_PATH}
cat ${PREPROCESSED_TRAIN_PATH} ${PREPROCESSED_TEST_PATH} > all_data.csv