"""
@author: Qinjuan Yang
@time: 2020-08-12 22:35
@desc: 
"""
import argparse
import os
import pickle as pkl

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch

from logger import get_logger
from model_utils import load_model_config

logger = get_logger()


def parse_arguments():
    parser = argparse.ArgumentParser(description="News Classification")

    # dataset
    parser.add_argument(
        "--train_path",
        type=str,
        help="path to the train dataset file"
    )

    # embedding
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="",
        help="pre-trained embedding file"
    )

    # model
    parser.add_argument(
        "--model",
        type=str,
        default="FastText",
        help="choose a model: TextCNN, TextRNN, FastText. default is TextCNN."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory to save model"
    )

    parser.add_argument(
        "--nsplits",
        type=int,
        default=10,
        help="number of splits of K-Fold."
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        help="Directory for configuration file."
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        help="Directory for log files."
    )

    args = parser.parse_args()
    logger.info("Arguments: {}".format(args))
    return args


if __name__ == '__main__':
    args = parse_arguments()

    config = load_model_config(args.config_dir, args.model)

    logger.info("Loading training dataset from file {}".format(args.train_path))
    train_df = pd.read_csv(args.train_path, sep="\t")
    train_df = train_df.sample(frac=1, axis=0)

    model_path = os.path.join(args.model_dir, "model_{}.model".format(args.model))

    if args.model == "FastText":
        from model_utils_fasttext import train_model

        # convert label to the format that fasttext needs
        train_df["label_ft"] = "__label__" + train_df["label"].astype("str")
        X_train, y_train = train_df["text"], train_df["label"]

        file_name = "train_ft.csv"

        logger.info("Start training FastText...")
        if args.nsplits > 1:
            SKF = StratifiedKFold(n_splits=args.nsplits, shuffle=True)
            for fold_idx, (train_idx, val_idx) in enumerate(SKF.split(X_train, y_train)):
                logger.info("*" * 20 + "Training {}-fold...".format(fold_idx))

                train_df[["text", "label_ft"]].iloc[train_idx].to_csv(file_name, header=None, index=None, sep=" ")
                train_model(file_name, config, X_val=X_train.iloc[val_idx].tolist(),
                            y_val=y_train.iloc[val_idx].tolist())
        else:
            train_df[["text", "label_ft"]].to_csv(file_name, header=None, index=False, sep=" ")
            train_model(file_name, config, save_model=True, model_path=model_path)

    else:
        from data_utils import build_vocab, build_dataset, build_iterator, load_embeddings
        from model_utils import load_model, train_model

        # split document to words
        logger.info("Split words...")
        train_df["text_words"] = train_df["text"].apply(lambda x: x.split())
        X_train, y_train = train_df["text_words"], train_df["label"]
        train_df.drop(columns=["text"], inplace=True)

        logger.info("Building dataset...")
        vocab2id = build_vocab(docs=X_train, min_count=config.min_count)
        pkl.dump(vocab2id, open(os.path.join(args.model_dir, "vocab_{}.vocab".format(args.model)), "wb"))
        train_data = build_dataset(X_train, vocab2id, max_doc_len=config.max_doc_len)
        train_df.drop(columns=["text_words"], inplace=True)

        logger.info("Loading embeddings...")
        embeddings = load_embeddings(args.embedding_path, vocab2id)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if args.nsplits > 1:
            SKF = StratifiedKFold(n_splits=args.nsplits, shuffle=True)
            for fold_idx, (train_idx, val_idx) in enumerate(SKF.split(X_train, y_train)):
                logger.info("*" * 20 + "Training {}-fold...".format(fold_idx))

                model = load_model(config, args.model, embeddings, embeddings.shape[1], len(y_train.unique()),
                                   device=device)

                train_iter = build_iterator(list(zip(train_data[train_idx], y_train[train_idx])), config.batch_size, device)
                val_iter = build_iterator(list(zip(train_data[val_idx], y_train[val_idx])), config.batch_size, device)

                train_model(config, model, args.log_dir, train_iter, val_iter, device)
        else:
            model = load_model(config, args.model, embeddings, embeddings.shape[1], len(y_train.unique()),
                               device=device)
            X_train_data, X_val_data, y_train_data, y_val_data = \
                train_test_split(train_data, y_train, test_size=0.25, stratify=y_train)
            train_iter = build_iterator(list(zip(X_train_data, y_train_data)), config.batch_size, device)
            val_iter = build_iterator(list(zip(X_val_data, y_val_data)), config.batch_size, device)

            train_model(config, model, args.log_dir, train_iter, val_iter, save_model=True, model_path=model_path)


