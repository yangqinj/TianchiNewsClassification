"""
@author: Qinjuan Yang
@time: 2020-08-12 23:04
@desc: 
"""
import time
from datetime import timedelta

from gensim import models
import numpy as np
from tqdm import tqdm
import torch

from logger import get_logger

logger = get_logger()


UNK, PAD = '<UNK>', '<PAD>'


def build_vocab(docs, min_count=1):
    logger.info("Start building vocabulary...")
    vocab = {
        PAD: min_count,
        UNK: min_count
    }
    for x, doc in enumerate(tqdm(docs)):
        for w in doc:
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1

    vocab2id = {w: x for x, (w, cnt) in enumerate(vocab.items()) if cnt >= min_count}
    logger.info("Finish building vocabulary with size {}".format(len(vocab2id)))

    return vocab2id


def build_dataset(docs, vocab2id, max_doc_len):
    """Convert word string to id."""
    logger.info("Start building dataset...")
    data = []
    for x, doc in enumerate(tqdm(docs)):
        if len(doc) > max_doc_len:
            doc = doc[:max_doc_len]
        else:
            doc.extend([PAD] * (max_doc_len - len(doc)))

        word_ids = [vocab2id.get(w, UNK) for w in doc]
        data.append(word_ids)

    logger.info("Finish building dataset with size {}".format(len(data)))

    return np.array(data)


def load_embeddings(embedding_file, vocab2id):
    """Load word embeddings to matrix."""
    logger.info("Loading embeddings...")

    word_vectors = models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    embed_size = word_vectors.vector_size

    embeddings = np.zeros((len(vocab2id), embed_size))
    for (word, wid) in vocab2id.items():
        try:
            embeddings[wid] = word_vectors.get_vector(word)
            embeddings[vocab2id.get(UNK)] += embeddings[wid]
        except KeyError as e:
            pass

    # set embedding for UNK as average of all word embeddings
    embeddings[vocab2id.get(UNK)] = embeddings[vocab2id.get(UNK)] / len(vocab2id)
    embeddings = embeddings / np.std(embeddings)

    logger.info("Finish loading embedding with size {}".format(embed_size))

    return torch.FloatTensor(embeddings)


class DatasetIterator(object):
    def __init__(self, data, batch_size, device, drop_last=False):
        self._data = data
        self._batch_size = batch_size
        self._device = device
        self._drop_last = drop_last
        self._residual = False
        self._index = 0

        self._n_batches = len(self._data) // self._batch_size
        if len(self._data) % self._batch_size != 0:
            if not self._drop_last:
                self._n_batches += 1
                self._residual = True

    def __len__(self):
        return self._n_batches

    def _to_tensor(self, datas):
        x = torch.LongTensor([d[0] for d in datas]).to(self._device)
        y = torch.LongTensor([d[1] for d in datas]).to(self._device)
        return x, y

    def __next__(self):
        if self._index == self._n_batches:
            self._index = 0
            raise StopIteration

        elif self._index == self._n_batches - 1:
            batch_data = self._data[self._index * self._batch_size:]
            self._index += 1
            batch_data = self._to_tensor(batch_data)
            return batch_data

        else:
            batch_data = self._data[self._index * self._batch_size: (self._index + 1) * self._batch_size]
            self._index += 1
            batch_data = self._to_tensor(batch_data)
            return batch_data

    def __iter__(self):
        return self


def build_iterator(data, batch_size, device):
    return DatasetIterator(data, batch_size, device)


def get_timedelta(start_time):
    end_time = time.time()
    cost_time = timedelta(seconds=(end_time - start_time))
    return cost_time








