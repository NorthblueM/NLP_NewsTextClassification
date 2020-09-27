# -*- coding: UTF-8 -*-

# @Date    : Sep 9, 2020
# @Author  : Nrothblue
"""训练流程"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

import globalvar as gl
seed = gl.get_value('seed')
np.random.seed(seed)

from dictionaries import *
from module_model import *
from trainer import *


def dataset_split(data_file, test_data_file, fold_num):
    """划分数据集"""

    # train data
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    # f = pd.read_csv(data_file, sep='\t', encoding='UTF-8', nrows=1000) # 小规模数据测试

    traincsv_texts = f['text'].tolist()
    traincsv_labels = f['label'].tolist()

    # 交叉验证数据集，随机采样
    fold_idx = {}
    kfold = StratifiedKFold(fold_num, shuffle=True, random_state=seed)

    for fold_i, [train_idx, val_idx] in enumerate(kfold.split(traincsv_texts, traincsv_labels)):

        logging.info("Fold id: %s, Train lens %s, Val lens %s", str(fold_i), str(len(train_idx)), str(len(val_idx)))
        # print(val_idx[:10])

        # shuffle
        np.random.seed(seed)
        np.random.shuffle(train_idx)
        np.random.seed(seed)
        np.random.shuffle(val_idx)

        fold_idx[fold_i] = {'train': train_idx, 'val': val_idx}

    # test data
    f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
    # f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8', nrows=100) # 小规模数据测试

    texts = f['text'].tolist()
    test_data = {'label': [0] * len(texts), 'text': texts}
    logging.info("Test lens %s", str(len(texts)))

    gl.set_value('test_data', test_data)

    return [traincsv_texts, traincsv_labels, fold_idx, test_data]



def train_flow(data_file, test_data_file, fold_num, run_fold, save_name, is_train=True):
    """训练全流程"""

    # 划分数据集
    [traincsv_texts, traincsv_labels, fold_idx, test_data] = dataset_split(data_file, test_data_file, fold_num)


    for fold_i in range(fold_num):
        if fold_i not in run_fold:
            continue

        logging.info("======Fold id: %s, Start Data Loader and Encoder", str(fold_i))
        train_idx = fold_idx[fold_i]['train']
        val_idx = fold_idx[fold_i]['val']

        labels = []
        texts = []
        for idx in train_idx:
            labels.append(traincsv_labels[idx])
            texts.append(traincsv_texts[idx])
        train_data = {'label': labels, 'text': texts}
    
        labels = []
        texts = []
        for idx in val_idx:
            labels.append(traincsv_labels[idx])
            texts.append(traincsv_texts[idx])
        dev_data = {'label': labels, 'text': texts}

        vocab = Vocab(train_data)
        model = Model(vocab)


        save_model = '../model/' + save_name + '_' + str(fold_i) + '.bin'
        save_test = '../user_data/' + save_name + '_' + str(fold_i) + '.csv'

        gl.set_value('train_data', train_data)
        gl.set_value('dev_data', dev_data)
        gl.set_value('save_model', save_model)
        gl.set_value('save_test', save_test)

        trainer = Trainer(model, vocab, is_train)

        # train
        if is_train:
            logging.info("======Fold id: %s, Start Training ", str(fold_i))
            trainer.train()

        # test
        logging.info("======Fold id: %s, Start Testing ", str(fold_i))
        trainer.test()
