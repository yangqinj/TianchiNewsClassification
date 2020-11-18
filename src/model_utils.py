"""
@author: Qinjuan Yang
@time: 2020-08-15 15:56
@desc: 
"""
import importlib
import json
import os
import time

from easydict import EasyDict as edict
import numpy as np
from sklearn import metrics
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from data_utils import get_timedelta

from logger import get_logger

logger = get_logger()


def load_model_config(config_dir, model_name):
    config = edict(json.load(open(os.path.join(config_dir, model_name + '.json'))))
    if not config:
        raise Exception("cannot read model config from directory {} "
                        "for model {}".format(config_dir, model_name))
    logger.info("Configuration: {}".format(config))
    return config


def load_model(config, model_name, embeddings, embed_size, num_classes, device):
    x = importlib.import_module('models.' + model_name)
    if not x:
        raise Exception("No model named {}".format(model_name))
    model = x.Model(embeddings,
                    embed_size,
                    num_classes,
                    config)
    model = model.to(device)
    return model


def class_accuracy(y_true, y_predict):
    matrix = metrics.confusion_matrix(y_true, y_predict)
    return matrix.diagonal() / matrix.sum(axis=1)


def cal_metrics(true, predict):
    acc = metrics.accuracy_score(true, predict)
    f1 = metrics.f1_score(true, predict, average='macro')
    class_acc = class_accuracy(true, predict)
    return acc, f1, class_acc


def cal_loss(true, predict):
    return metrics.log_loss(true, predict)


def evaluate_model(model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for datas, labels in data_iter:
            model.eval()
            outputs = model(datas)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            predict_all = np.append(predict_all, torch.max(outputs.data, 1)[1].cpu().numpy())
            labels_all = np.append(labels_all, labels.data.cpu().numpy())

    return loss_total/len(data_iter), cal_metrics(labels_all, predict_all)


def train_model(config, model, log_dir, train_iter, dev_iter, save_model=False, model_path=None):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.learning_rate_decay_step,
                                                gamma=config.learning_rate_decay_rate)
    num_steps = 0
    dev_best_loss = float('inf')
    dev_best_step = 0

    writer = SummaryWriter(os.path.join(log_dir,
                                        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))

    try:
        for epoch in range(config.epochs):
            for x, (datas, labels) in enumerate(train_iter):
                num_steps += 1
                outputs = model(datas)
                loss = F.cross_entropy(outputs, labels)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                if num_steps % config.log_interval == 0:
                    true = labels.data.cpu().numpy()
                    predict = torch.max(outputs.data, 1)[1].cpu().numpy()

                    train_acc, train_f1, train_class_acc = cal_metrics(true, predict)
                    dev_loss, (dev_acc, dev_f1, dev_class_acc) = evaluate_model(model, dev_iter)

                    msg = 'Epoch: {:>4}/{:<4} Iter: {:>4}/{:<4}, \ntrain loss = {:>6.4f}, ' \
                          'acc: {:>6.4f}, f1: {:>6.4f}, class_acc: {}, ' \
                          '\ndev loss = {:>6.4f}, acc = {:>6.4f}, f1: {:>6.4f}, ' \
                          'class_acc: {}, Time: {}'
                    logger.info(msg.format(epoch, config.epochs, x, len(train_iter),
                                           loss, train_acc, train_f1, ", ".join(["{:.4f}".format(acc) for acc in train_class_acc]),
                                           dev_loss, dev_acc, dev_f1, ", ".join(["{:.4f}".format(acc) for acc in dev_class_acc]),
                                           get_timedelta(start_time)))

                    writer.add_scalar("train/loss", loss.item(), num_steps)
                    writer.add_scalar("train/acc", train_acc, num_steps)
                    writer.add_scalar("train/f1", train_f1, num_steps)
                    class_labels = ['class_' + str(x) for x in range(len(train_class_acc))]
                    writer.add_scalars("train/class_acc",
                                       dict(zip(class_labels, train_class_acc)),
                                       num_steps)

                    writer.add_scalar("dev/loss", dev_loss.item(), num_steps)
                    writer.add_scalar("dev/acc", dev_acc, num_steps)
                    writer.add_scalar("dev/f1", dev_f1, num_steps)
                    class_labels = ['class_' + str(x) for x in range(len(dev_class_acc))]
                    writer.add_scalars("dev/class_acc",
                                       dict(zip(class_labels, dev_class_acc)),
                                       num_steps)

                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        dev_best_step = num_steps
                    elif num_steps - dev_best_step > config.early_stop_step:
                        raise StopIteration('early stop for {} steps'.format(config.early_stop_step))

                    model.train()
                scheduler.step()

    except StopIteration as e:
        logger.error(e)

    writer.close()

    if save_model and model_path:
        logger.info("Saving model to {}".format(model_path))
        torch.save(model, model_path)


def test_model(model, data_iter):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch_data in data_iter:
            outputs = model(batch_data)
            batch_pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            y_pred = y_pred + list(batch_pred)

    return y_pred

