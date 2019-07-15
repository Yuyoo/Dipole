# -*- coding: utf-8 -*-
# @Time    : 2019-07-08 14:01
# @Author  : Yuyoo
# @Email   : sunyuyaoseu@163.com
# @File    : data2bow.py

import pandas as pd
from tqdm import tqdm


def parse_series_lst(x):
    """
    将series列表文本转换为列表
    :param x:
    :return:
    """
    lst = x.strip('\[|\]').split(' ')
    lst = [eval(s.strip('\\n')) for s in lst]
    return lst


def generate_map_dict(data):
    """
    生成映射字典表
    :param data:
    :return:
    """
    diag_dict_lst = []
    drug_dict_lst = []
    for i, sid in enumerate(tqdm(data.SUBJECT_ID.unique())):
        s_icd = data[data.SUBJECT_ID == sid].ICD10.tolist()
        s_atc = data[data.SUBJECT_ID == sid].atc.tolist()
        for l in s_icd:
            l = parse_series_lst(l)
            for x in l:
                diag_dict_lst.append(x)

        for l in s_atc:
            l = parse_series_lst(l)
            for x in l:
                drug_dict_lst.append(x)

    diag_dict_lst = list(set(diag_dict_lst))
    drug_list_lst = list(set(drug_dict_lst))
    print("数据中共有诊断{}种".format(len(diag_dict_lst)))
    print("数据中共有药物{}种".format(len(drug_list_lst)))

    all_dict_list = drug_list_lst + diag_dict_lst
    all_dict = {all_dict_list[i]: i for i in range(len(all_dict_list))}
    return all_dict


def data2bow(raw_data_file):
    data = pd.read_csv(raw_data_file)
    data.ADMITTIME = pd.to_datetime(data.ADMITTIME)
    data.sort_values(by=['SUBJECT_ID', 'ADMITTIME'], inplace=True)
    # 生成映射字典表
    all_dict = generate_map_dict(data)
    # 转换成retain输入格式
    all_lst = []
    for i, sid in enumerate(tqdm(data.SUBJECT_ID.unique())):
        if i != 0:
            all_lst.append('-1')
        s_atc = data[data.SUBJECT_ID == sid].atc.tolist()
        s_icd = data[data.SUBJECT_ID == sid].ICD10.tolist()

        for l in range(len(s_atc)):
            l_atc = parse_series_lst(s_atc[l])
            l_icd = parse_series_lst(s_icd[l])
            l_atc = l_atc + l_icd
            atc_idx = ''
            for j, atc in enumerate(l_atc):
                if j == 0:
                    atc_idx += str(all_dict[atc]) + ':1'
                else:
                    atc_idx += ',' + str(all_dict[atc]) + ':1'
            all_lst.append(atc_idx)
    return all_lst


if __name__ == '__main__':
    raw_data_file = './data.csv'
    all_lst = data2bow(raw_data_file)
    with open('../pat_all_lst.txt', 'w') as f:
        for l in all_lst:
            f.write(l + '\n')
