"""
@author: Qinjuan Yang
@time: 2020-08-15 16:15
@desc: Train embeddings from given documents.
"""
import argparse
import time
from datetime import timedelta
import os

from gensim.models import Word2Vec


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str
    )

    parser.add_argument(
        "--size",
        type=int,
        default="size of word embedding"
    )

    parser.add_argument(
        "--embedding_path",
        type=str,
        default="path for training embeddings"
    )

    args = parser.parse_args()

    start_time = time.time()

    model = Word2Vec(corpus_file=args.dataset_path, size=args.size)
    model.wv.save_word2vec_format(args.embedding_path, binary=True)

    end_time = time.time()
    print('Time usage for training:', timedelta(seconds=(end_time - start_time)))
