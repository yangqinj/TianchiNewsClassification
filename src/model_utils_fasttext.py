"""
@author: Qinjuan Yang
@time: 2020-10-29 21:01
@desc: 
"""
import fasttext


from logger import get_logger
from model_utils import cal_metrics

logger = get_logger()


def train_model(filename, config, X_val=None, y_val=None, save_model=False, model_path=None):
    model = fasttext.train_supervised(input=filename,
                                      lr=config.learning_rate,
                                      dim=config.embedding_dim,
                                      epoch=config.epochs,
                                      wordNgrams=config.word_ngrams,
                                      loss=config.loss,
                                      minCount=config.min_count)

    # test on validation dataset
    if X_val and y_val:
        y_pred = [int(model.predict(d)[0][0].split('_')[-1]) for d in X_val]
        acc, f1, class_acc = cal_metrics(y_val, y_pred)
        logger.info("[validate] acc={:.4f} f1={:.4f} class_acc={}".format(acc, f1, class_acc))

    # save model for testing
    if save_model and model_path:
        logger.info("Saving model to {}".format(model_path))
        model.save_model(model_path)


def test_model(model, X_test):
    y_pred = [int(model.predict(d)[0][0].split('_')[-1]) for d in X_test]
    return y_pred
