# -*- coding: UTF-8 -*-

# @Date    : Sep 9, 2020
# @Author  : Nrothblue
"""伪标签生成"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

import pandas as pd

def pseudo_label(save_name, run_fold, data_file, test_file, data_file_pseudo):
    """伪标签生成"""

    save_tests = {}
    for fold_i in run_fold:
        save_test = '../user_data/' + save_name + '_' + str(fold_i) + '.csv'
        save_tests[fold_i] = save_test


    df_merge = pd.DataFrame()
    for fold_i in run_fold:
        df = pd.read_csv(save_tests[fold_i])
        col = 'label_'+str(fold_i)
        df_merge[col] = df['label']
    df_merge.to_csv('../user_data/' + save_name + '_merge.csv', index=None)

    # 投票
    df_vote = pd.DataFrame()
    df_vote['label'] = df_merge.apply(lambda x:x.value_counts().idxmax(), axis=1)
    df_vote.to_csv('../user_data/' + save_name + '_vote.csv', index=None)

    # 可信度评估
    df_look = pd.DataFrame()
    df_look = df_merge
    df_look['vote'] = df_vote['label']

    def is_all_same(ser):
        for idx in ser.index:
            if ser.iloc[0] != ser.loc[idx]:
                return 0
        return 1
    df_look['all_same'] = df_look.apply(is_all_same, axis=1)


    # 伪标签数据生成
    traincsv_data = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    testcsv_data = pd.read_csv(test_file, sep='\t', encoding='UTF-8')
    # traincsv_data = pd.read_csv(data_file, sep='\t', encoding='UTF-8', nrows=1000)
    # testcsv_data = pd.read_csv(test_file, sep='\t', encoding='UTF-8', nrows=100)

    test_weaklabel_same = pd.DataFrame()
    test_weaklabel_same['label'] = df_vote[df_look['all_same'] == 1]['label']
    test_weaklabel_same['text'] = testcsv_data[df_look['all_same'] == 1]['text']

    traincsv_weaklabel_same = pd.concat([traincsv_data, test_weaklabel_same]).reset_index()
    del traincsv_weaklabel_same['index']
    traincsv_weaklabel_same.to_csv(data_file_pseudo, index=None, sep='\t')

    logging.info("Pseudo_label_num: %s", str(test_weaklabel_same.shape[0]))
    logging.info("New_train_data_num: %s", str(traincsv_weaklabel_same.shape[0]))
