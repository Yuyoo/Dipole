# -*- coding: utf-8 -*-
# @Time    : 2019-07-07 10:53
# @Author  : Yuyoo
# @Email   : sunyuyaoseu@163.com
# @File    : Dipole_torch.py


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np


class Dipole(nn.Module):
    def __init__(self, input_dim, day_dim, output_dim, rnn_hiddendim, keep_prob=1.0):
        super(Dipole, self).__init__()
        self.input_dim = input_dim
        self.day_dim = day_dim
        self.output_dim = output_dim
        self.rnn_hiddendim = rnn_hiddendim
        self.keep_prob = keep_prob
        # self.opt = opt
        # self.L2 = L2

        self.day_embedding = nn.Linear(self.input_dim, self.day_dim)
        self.dropout = nn.Dropout(self.keep_prob)
        self.attn = nn.Linear(self.rnn_hiddendim * 2, 1)
        self.gru = nn.GRU(self.day_dim, self.rnn_hiddendim)
        self.gru_reverse = nn.GRU(self.day_dim, self.rnn_hiddendim)
        self.attn_out = nn.Linear(self.rnn_hiddendim * 4, self.day_dim)
        self.out = nn.Linear(self.day_dim, self.output_dim)

    def attentionStep(self, h_0, att_timesteps):
        """
        根据前att_timesteps步，经gru，attention，得到当前时间步的输出
        :param h_0: 隐藏层初始状态
        :param att_timesteps: 当前时间步
        :return: 当前时间步的隐藏层状态输出
        """
        day_emb = self.day_emb[:att_timesteps]  # 取出前n-1步输入
        rnn_h = self.gru(day_emb, h_0)[0]
        day_emb_reverse = self.day_emb[:att_timesteps].flip(dims=[0])   # 取出前n-1步，反向
        rnn_h_reverse = self.gru_reverse(day_emb_reverse, h_0)[0]

        rnn_h = torch.cat((rnn_h, rnn_h_reverse), 2)    # 按特征维度进行拼接，shape=(seq_len, batch_size, 2*hidden_size)

        Alpha = self.attn(rnn_h)    # 线性降维，shape=(seq_len, batch_size, 1)
        Alpha = torch.squeeze(Alpha, dim=2)     # 消除多余维度，shape=(seq_len, batch_size)
        Alpha = torch.transpose(F.softmax(torch.transpose(Alpha, 0, 1)), 0, 1)  # softmax获得attention的值

        attn_applied = Alpha.unsqueeze(2) * rnn_h   # 增加维度，使alpha可以与rnn相乘, shape=(seq_len, batch_size, 2*hidden_size)
        c_t = torch.mean(attn_applied, 0)   # 按时间维度聚合，shape=(batch_size, 2*hidden_size)
        h_t = torch.cat((c_t, rnn_h[-1]), dim=1)    # 按特征维度拼接，shape=(batch_size, 4*hidden_size)

        h_t_out = self.attn_out(h_t)    # attention输出降维，shape=(batch_size, day_dim)
        return h_t_out

    def forward(self, x):
        # x = torch.tensor(x)
        # embedding层
        h_0 = self.initHidden(x.shape[1])
        self.day_emb = self.day_embedding(x)    # shape=(seq_len, batch_size, day_dim)

        # LSTM层
        if self.keep_prob < 1.0:
            self.day_emb = self.dropout(self.day_emb)

        count = np.arange(x.shape[0]) + 1
        h_t_out = torch.zeros_like(self.day_emb)    # shape=(seq_len, batch_size, day_dim)
        for i, att_timesteps in enumerate(count):
            # 按时间步迭代，计算每个时间步的经attention的gru输出
            h_t_out[i] = self.attentionStep(h_0, att_timesteps)

        # output层
        y_hat = self.out(h_t_out)   # shape=(seq_len, batch_size, out_dim)
        y_hat = torch.sigmoid(y_hat)    # shape=(seq_len, batch_size, out_dim)

        return y_hat

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.rnn_hiddendim, device=torch.device('cuda:1'))

    def padTrainMatrix(self, seqs):
        lengths = np.array([len(seq) for seq in seqs]).astype("int32")
        n_samples = len(seqs)
        maxlen = np.max(lengths)

        x = np.zeros([maxlen, n_samples, self.input_dim]).astype(np.float32)
        for idx, seq in enumerate(seqs):
            for xvec, subseq in zip(x[:, idx, :], seq):
                for tuple in subseq:
                    xvec[tuple[0]] = tuple[1]
        return x, lengths
