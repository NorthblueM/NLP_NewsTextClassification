# -*- coding: UTF-8 -*-

# @Date    : Sep 9, 2020
# @Author  : Nrothblue
"""模型融合集成"""

import pandas as pd

def vote_weight(save_names, run_folds, weights, pred_name):
    """加权投票"""

    def get_file_name(save_names, fold_ids):
        save_tests = []
        for i in range(len(save_names)):
            save_name = save_names[i]
            for fold_i in fold_ids[i]:
                save_test = '../user_data/' + save_name + '_' + str(fold_i) + '.csv'
                save_tests.append(save_test)

        return save_tests

    save_tests = get_file_name(save_names, run_folds)

    file_name = '-'.join(save_names)

    df_merge = pd.DataFrame()
    for save_test in save_tests:
        df = pd.read_csv(save_test)
        df_merge[save_test] = df['label']

    df_merge.to_csv('../user_data/' + file_name + '-merge.csv', index=None)


    def vote_w(ser):
        group_cols_ls = []
        for name in save_names:
            cols_ls = []
            for col in ser.index:
                if name in col:
                    cols_ls.append(col)
            group_cols_ls.append(cols_ls)

        group_value_counts = []
        for i, cols_ls in enumerate(group_cols_ls):
            group_value_counts.append(ser[cols_ls].value_counts() * weights[i])

        for i, count in enumerate(group_value_counts):
            if i == 0:
                value_count = group_value_counts[0]
            else:
                value_count = value_count.add(count, fill_value=0)

        return value_count.idxmax()

    df_vote = pd.DataFrame()

    df_vote['label'] = df_merge.apply(lambda x:x.value_counts().idxmax(), axis=1)
    df_vote.to_csv('../user_data/' + file_name + '-vote.csv', index=None)

    df_vote['label'] = df_merge.apply(vote_w, axis=1)
    df_vote.to_csv('../user_data/' + file_name + '-vote_wight.csv', index=None)
    df_vote.to_csv('../prediction_result/' + pred_name, index=None)
