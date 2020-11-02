"""
@author: Qinjuan Yang
@time: 2020-10-28 21:03
@desc: 
"""
import argparse
import os
import pandas as pd
import torch

from logger import get_logger
from model_utils import load_model_config

logger = get_logger()


def parse_arguments():
    parser = argparse.ArgumentParser(description="News Classification")

    # dataset
    parser.add_argument(
        "--test_path",
        type=str,
        help="path to the test dataset file"
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info("Loading data...")
    test_df = pd.read_csv(args.test_path)
    logger.info("Finish loading data with size {}".format(test_df.shape[0]))

    model_path = os.path.join(args.model_dir, "model_{}.model".format(args.model))

    if args.model == "FastText":
        import fasttext
        from model_utils_fasttext import test_model

        logger.info("Loading model from {}".format(model_path))
        model = fasttext.load_model(model_path)

        logger.info("Testing model...")
        y_pred = test_model(model, X_test=test_df["text"].tolist())
    else:
        from data_utils import build_vocab, build_dataset
        from model_utils import test_model

        test_df["text_words"] = test_df["text"].apply(lambda x: x.split())
        vocab2id = build_vocab(test_df["text_words"], min_count=config.min_count)
        data = build_dataset(test_df["text_words"], vocab2id, max_doc_len=config.max_doc_len)

        logger.info("Loading model from {}".format(model_path))
        model = torch.load(model_path)

        logger.info("Testing model...")
        y_pred = test_model(model, torch.LongTensor(data), device)

    test_df["label"] = y_pred
    test_df["label"].to_csv("predict_{}.csv".format(args.model), index=False, header="label")
    logger.info("Finishing predict label for testing data.")

