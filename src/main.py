# -*- coding: UTF-8 -*-

# @Date    : Sep 9, 2020
# @Author  : Nrothblue
"""程序入口"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

import globalvar as gl
seed = 666
gl._init()
gl.set_value('seed', seed)

import random
import numpy as np
import torch


from train_flow import *
from label_test import *
from ensemble import *

def _main():

    # 模型和预测结果输出名称
    save_names = ['model_trainset_pseudo0', 'model_trainset_pseudoa', 'model_trainset_pseudob']
    # 训练折数
    fold_nums = [10, 10, 10]
    # 要训练的折
    run_folds = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 3, 6, 9]]
    # 模型权重
    weights = [0.9*2, 0.9*2.5, 0.9*3*2.5]

    # 小规模数据测试
    # # 模型和预测结果输出名称
    # save_names = ['model_trainset_pseudo0', 'model_trainset_pseudoa', 'model_trainset_pseudob']
    # # 训练折数
    # fold_nums = [10, 10, 10]
    # # 要训练的折
    # run_folds = [[0, 1], [0, 1], [0]]
    # # 模型权重
    # weights = [0.9*2, 0.9*2.5, 0.9*3*2.5]


    logging.info('============Semi-Supervised Train 1 (for test_a).')
    # 第一次半监督训练
    data_file = '../data/train_set.csv'
    train_data_file = data_file
    test_data_file = '../data/test_a.csv'
    fold_num = fold_nums[0]
    run_fold = run_folds[0]
    save_name = save_names[0]
    semi_a_times = 1
    for semi_i in range(semi_a_times):
        train_flow(train_data_file, test_data_file, fold_num, run_fold, save_name, is_train=True)
        data_file_pseudo = '../user_data/' + 'train_set_pseudo_a.csv'
        pseudo_label(save_name, run_fold, data_file, test_data_file, data_file_pseudo)
        train_data_file = data_file_pseudo

    logging.info('============Semi-Supervised Train 2 (for test_b).')
    # 第二次半监督训练
    data_file = '../user_data/train_set_pseudo_a.csv'
    train_data_file = data_file
    test_data_file = '../data/test_b.csv'
    fold_num = fold_nums[1]
    run_fold = run_folds[1]
    save_name = save_names[1]
    semi_b_times = 1
    for semi_i in range(semi_b_times):
        train_flow(train_data_file, test_data_file, fold_num, run_fold, save_name, is_train=True)
        data_file_pseudo = '../user_data/' + 'train_set_pseudo_b.csv'
        pseudo_label(save_name, run_fold, data_file, test_data_file, data_file_pseudo)
        train_data_file = data_file_pseudo


    logging.info('============Last Train (for test_b).')
    # 最后一次数据再次训练
    data_file = '../user_data/train_set_pseudo_b.csv'
    test_data_file = '../data/test_b.csv'
    fold_num = fold_nums[2]
    run_fold = run_folds[2]
    save_name = save_names[2]
    train_flow(data_file, test_data_file, fold_num, run_fold, save_name, is_train=True)


    logging.info('============No Pseudo Test (for test_b).')
    # 预测未加伪标签数据的模型的test_b的结果
    data_file = '../data/train_set.csv'
    test_data_file = '../data/test_b.csv'
    fold_num = fold_nums[0]
    run_fold = run_folds[0]
    save_name = save_names[0]
    train_flow(data_file, test_data_file, fold_num, run_fold, save_name, is_train=False)


    logging.info('============Model Ensemble, Predicted Results Vote.')
    # 模型集成融合
    pred_name = 'predictions.csv'
    vote_weight(save_names, run_folds, weights, pred_name)



if __name__ == "__main__":

    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    # set cuda
    gpu = 0
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
    else:
        device = torch.device("cpu")

    logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

    # 设置全局变量
    gl.set_value('use_cuda', use_cuda)
    gl.set_value('device', device)

    # 开始完整流程
    _main()
