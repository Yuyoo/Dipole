# -*- coding: utf-8 -*-
# @Time    : 2019-07-06 23:56
# @Author  : Yuyoo
# @Email   : sunyuyaoseu@163.com
# @File    : Dipole_tf_Runner.py


import numpy as np
import time
from dipole_tf.Dipole_tf import Dipole
import math
from operator import itemgetter
from utils import metric_report

starttime = time.time()


# 读取数据
def read_data(d2bow_file, linesize=0):
    lasttime = time.time()
    with open(d2bow_file, 'r') as f:
        d2bow = []
        d2v = []
        mask = []
        daynum = 0
        visitnum = 1
        linenum = 0
        for line in f.readlines():
            linenum += 1
            if linesize != 0 and linenum >= linesize:
                break
            db = []
            if line.strip() == '-1':
                d2bow.append(-1)
                d2v.append(0)
                visitnum += 1
                mask.append([0])
            else:
                pairs = line.strip().split(',')
                for pair in pairs:
                    pw = int(pair.split(':')[0])
                    pc = float(pair.split(':')[1])
                    tu = (pw, pc)
                    db.append(tu)
                d2bow.append(db)
                daynum += 1
                d2v.append(visitnum)
                mask.append([1])

            if linenum % 100000 == 0:
                print('finish reading line %d, takes %f' % (linenum, time.time() - lasttime))
                lasttime = time.time()

    d2v = np.array(d2v).astype('int32')
    mask = np.array(mask).astype('float32')
    print('len(days): %d' % (len(d2v)))
    print('day size: %d' % (daynum))
    print('visit size: %d' % (visitnum))

    return d2bow, d2v, mask


def getTrainData(num):
    filename = "./pat_all_lst.txt"
    d2bow, d2v, mask = read_data(filename, num)

    temp_patient = []
    patients = []
    for list in d2bow:
        if list == -1:
            patients.append(temp_patient)
            temp_patient = []
        else:
            temp_patient.append(list)
    if len(temp_patient) != 0:
        patients.append(temp_patient)

    print(len(patients))
    return patients


def train_predict(batch_size=100, epochs=10, topk=30):
    patients = getTrainData(4000000)  # patients × visits × medical_code

    patients_num = len(patients)
    train_patient_num = int(patients_num * 0.8)
    patients_train = patients[0:train_patient_num]
    test_patient_num = patients_num - train_patient_num
    patients_test = patients[train_patient_num:]

    train_batch_num = int(np.ceil(float(train_patient_num) / batch_size))
    test_batch_num = int(np.ceil(float(test_patient_num) / batch_size))

    retain = Dipole(input_dim=3393, day_dim=200, output_dim=283, rnn_hiddendim=300)
    for epoch in range(epochs):
        starttime = time.time()
        # 训练
        all_loss = 0.0
        for batch_index in range(train_batch_num):
            patients_batch = patients_train[batch_index * batch_size:(batch_index + 1) * batch_size]
            patients_batch_reshape, patients_lengths = retain.padTrainMatrix(
                patients_batch)  # maxlen × n_samples × inputDimSize
            batch_x = patients_batch_reshape[0:-1]  # 获取前n-1个作为x，来预测后n-1天的值
            # batch_y = patients_batch_reshape[1:]
            batch_y = patients_batch_reshape[1:, :, :283]

            loss, y_hat = retain.startTrain(batch_x, batch_y)
            all_loss += loss
        print("Train:Epoch-" + str(epoch) + ":" + str(all_loss) + " Train Time:" + str(time.time() - starttime))

        # 测试
        NDCG = 0.0
        RECALL = 0.0
        DAYNUM = 0.0
        all_loss = 0.0
        gbert_pred = []
        gbert_true = []
        gbert_len = []

        for batch_index in range(test_batch_num):
            patients_batch = patients_test[batch_index * batch_size:(batch_index + 1) * batch_size]
            patients_batch_reshape, patients_lengths = retain.padTrainMatrix(patients_batch)
            batch_x = patients_batch_reshape[0:-1]
            # batch_y = patients_batch_reshape[1:]
            batch_y = patients_batch_reshape[1:, :, :283]

            loss, y_hat = retain.get_results(batch_x, batch_y)
            all_loss += loss
            ndcg, recall, daynum = validation(y_hat, patients_batch, patients_lengths, topk)
            # acc_container = metric_report(y_hat, patients_batch)
            NDCG += ndcg
            RECALL += recall
            DAYNUM += daynum
            gbert_pred.append(y_hat)
            gbert_true.append(batch_y)
            gbert_len.append(patients_lengths)

        avg_NDCG = NDCG / DAYNUM
        avg_RECALL = RECALL / DAYNUM
        y_pred_all, y_true_all = batch_squeeze(gbert_pred, gbert_true, gbert_len)
        acc_container = metric_report(y_pred_all, y_true_all, 0.2)
        print("Test:Epoch-" + str(epoch) + " Loss:" + str(all_loss) + " Test Time:" + str(time.time() - starttime))
        print("Test:Epoch-" + str(epoch) + " NDCG:" + str(avg_NDCG) + " RECALL:" + str(avg_RECALL))
        print("Test:Epoch-" + str(epoch) + " Jaccard:" + str(acc_container['jaccard']) +
              " f1:" + str(acc_container['f1']) + " prauc:" + str(acc_container['prauc']) + " roauc:" + str(
            acc_container['auc']))

        print("")


def batch_squeeze(gbert_pred, gbert_true, gbert_len):
    y_pred_all = []
    y_true_all = []
    for b in range(len(gbert_len)):
        y_pred = np.transpose(gbert_pred[b], (1, 0, 2))
        y_true = np.transpose(gbert_true[b], (1, 0, 2))
        v_len = gbert_len[b]
        for p in range(y_pred.shape[0]):
            for v in range(v_len[p] - 1):
                y_pred_all.append(y_pred[p][v].reshape(1, -1))
                y_true_all.append(y_true[p][v].reshape(1, -1))
    y_pred_all = np.concatenate(y_pred_all)
    y_true_all = np.concatenate(y_true_all)
    return y_pred_all, y_true_all


# 根据训练的y_hat和y进行topk的计算
# 这里的y_true是没有经过padTrainMatrix的实际输入
def validation(y_hat, y_true, length, topk):
    # 将维度改变为 maxlen × n_samples × outputDimSize
    y_hat = np.transpose(y_hat, (1, 0, 2))

    NDCG = 0.0
    RECALL = 0.0
    daynum = 0

    n_patients = y_hat.shape[0]
    for i in range(n_patients):
        predict_one = y_hat[i]
        y_true_one = y_true[i]
        len_one = length[i]

        # 减1是因为预测的是第2~n天，不包含第一天
        for i in range(len_one - 1):
            y_pred_day = predict_one[i]
            y_true_day = y_true_one[i + 1]
            ndcg, lyt, ltp = evaluate_predict_performance(y_pred_day.flatten(), y_true_day, topk)
            NDCG += ndcg
            recall = 0.0
            if lyt != 0:
                recall += ltp * 1.0 / lyt
            else:
                recall += 1.0
            RECALL += recall
            daynum += 1

    return NDCG, RECALL, daynum


# 计算每一天的topk
def evaluate_predict_performance(y_pred, y_bow_true, topk=30):
    sorted_idx_y_pred = np.argsort(-y_pred)

    if topk == 0:
        sorted_idx_y_pred_topk = sorted_idx_y_pred[:len(y_bow_true)]
    else:
        sorted_idx_y_pred_topk = sorted_idx_y_pred[:topk]

    sorted_bow = sorted(y_bow_true, key=itemgetter(1), reverse=True)
    sorted_idx_y_true = []
    for sb in sorted_bow:
        if sb[1] > 1e-3:
            sorted_idx_y_true.append(sb[0])

    # print(sorted_idx_y_pred_topk)
    # print(sorted_idx_y_true)

    true_part = set(sorted_idx_y_true).intersection(set(sorted_idx_y_pred_topk))  # 重合部分，用来计算ndcg
    idealDCG = 0.0
    for i in range(len(sorted_idx_y_true)):
        idealDCG += (2 ** 1 - 1) / math.log(1 + i + 1)

    DCG = 0.0
    for i in range(len(sorted_idx_y_true)):
        if sorted_idx_y_true[i] in true_part:
            DCG += (2 ** 1 - 1) / math.log(1 + i + 1)

    # print('true lab size: %d, intersection part size: %d' %(len(sorted_idx_y_true), len(true_part)))
    if idealDCG != 0:
        NDCG = DCG / idealDCG
    else:
        NDCG = 1
    # print('NDCG: ' + str(NDCG))
    return NDCG, len(sorted_idx_y_true), len(true_part)


if __name__ == "__main__":
    # patients = getTrainData(100)
    #
    # re = Retain(inputDimSize=4216,embDimSize=300,alphaHiddenDimSize=200,betaHiddenDimSize=200,outputDimSize=4216)
    # patients,length = re.padTrainMatrix(patients)
    # X = patients[0:-1]
    # Y = patients[1:]
    # re.startTrain(X,Y,length)

    train_predict(epochs=10)
