#coding:utf-8
import tensorflow as tf
import numpy as np

class Dipole(object):

    def __init__(self,input_dim, day_dim,output_dim,rnn_hiddendim,keep_prob = 1.0,L2=1e-8,
                 opt=tf.train.AdadeltaOptimizer(learning_rate=10), init_scale=0.01):

        self.input_dim = input_dim
        self.day_dim = day_dim
        self.output_dim = output_dim
        self.rnn_hiddendim = rnn_hiddendim
        self.keep_prob = keep_prob
        self.opt = opt
        self.L2 = L2
        self.init_scale = init_scale

        self.tparams = {}

        #第一层降维层
        with tf.variable_scope("Day_embedding") as scope:
            self.tparams["input_w"] = tf.get_variable(shape=[self.input_dim,self.day_dim],dtype=tf.float32,name="input_w")
            self.tparams["input_b"] = tf.get_variable(shape=[self.day_dim],dtype=tf.float32,name="input_b")

        # RNN层 使用LSTM
        with tf.variable_scope("RNN") as scope:
            self.tparams["W_gru"] = tf.get_variable(shape=[self.day_dim,4*self.rnn_hiddendim],name="W_gru",dtype=tf.float32)
            self.tparams["U_gru"] = tf.get_variable(shape=[self.rnn_hiddendim,4*self.rnn_hiddendim],name="U_gru",dtype=tf.float32)
            self.tparams["b_gru"] = tf.get_variable(shape=[4*self.rnn_hiddendim],name="b_gru",dtype=tf.float32)


        # RNN层  使用LSTM  逆序
        with tf.variable_scope("RNN_Reverse") as scope:
            self.tparams["W_gru_reverse"] = tf.get_variable(shape=[self.day_dim, 4 * self.rnn_hiddendim],
                                                            name="W_gru_reverse",
                                                            dtype=tf.float32)
            self.tparams["U_gru_reverse"] = tf.get_variable(shape=[self.rnn_hiddendim, 4* self.rnn_hiddendim],
                                                            name="U_gru_reverse", dtype=tf.float32)
            self.tparams["b_gru_reverse"] = tf.get_variable(shape=[4 * self.rnn_hiddendim], name="b_gru_reverse",
                                                            dtype=tf.float32)

        # 最后一层输出
        with tf.variable_scope("output") as scope:
            self.tparams["output_w"] = tf.get_variable(shape=[self.day_dim,self.output_dim],name="output_w",dtype=tf.float32)
            self.tparams["output_b"] = tf.get_variable(shape=[self.output_dim],name="output_b",dtype=tf.float32)


        # RNN输出后的Attention
        with tf.variable_scope("Attention") as scope:
            self.tparams["w_alpha"] = tf.get_variable(shape=[self.rnn_hiddendim*2 , 1], name="w_alpha",
                                                      dtype=tf.float32)
            self.tparams["b_alpha"] = tf.get_variable(shape=[1, 1], name="b_alpha", dtype=tf.float32)

            self.tparams["w_beta"] = tf.get_variable(shape=[self.rnn_hiddendim*2, self.day_dim], name="w_beta",
                                                     dtype=tf.float32)
            self.tparams["b_beta"] = tf.get_variable(shape=[self.day_dim], name="b_beta", dtype=tf.float32)

            self.tparams["W_c"] = tf.get_variable(shape=[self.rnn_hiddendim*4,self.day_dim],name="W_c",dtype=tf.float32)

        # 输入数据与Label数据
        self.x = tf.placeholder(tf.float32, [None, None, self.input_dim],
                                name="input")  # 输入数据格式为[maxlen × patients(n_samples) × input_dim]
        self.y = tf.placeholder(tf.float32, [None, None, self.output_dim], name="label")  # 输出数据与输入基本相同，两者相差一天

        self.first_ct = tf.placeholder(tf.float32,[None,self.day_dim],name="first_ch") #RNN网络中初始化的c_t
        self.first = tf.placeholder(tf.float32,[2,None,self.rnn_hiddendim],name="first")
        self.d2diag = tf.placeholder(tf.int32,[None,None,2],name="d2diag") #  maxlen × patients_num × 2 大小的矩阵，每天对应的diag编号
        self.gap = tf.placeholder(tf.int32,[None,None,None,2],name="gap")   # maxlen × patients_num × maxgap × 2
        self.counts = tf.placeholder(tf.int32,[None],name="counts")

        self.loss = self.build_net()
        self.opt = self.opt.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _slice(self,x,n,dim):
        return x[:,n*dim:(n+1)*dim]


    def gru_layer(self,day_emb,hiddenDimSize,reverse = False):

        if reverse == True:
            U_gru = self.tparams["U_gru_reverse"]
            W_gru = self.tparams["W_gru_reverse"]
            b_gru = self.tparams["b_gru_reverse"]
        else:
            U_gru = self.tparams["U_gru"]
            W_gru = self.tparams["W_gru"]
            b_gru = self.tparams["b_gru"]

        #按照时间顺序得到ht~h1
        def stepFn(h_ct,wx):

            h = h_ct[0]
            ct = h_ct[1]

            uh = tf.matmul(h,U_gru)

            f = tf.nn.sigmoid( self._slice(wx,0,hiddenDimSize) + self._slice(uh,0,hiddenDimSize) )
            i = tf.nn.sigmoid( self._slice(wx,1,hiddenDimSize) + self._slice(uh,1,hiddenDimSize) )
            c = tf.nn.tanh( self._slice(wx,2,hiddenDimSize) + self._slice(uh,2,hiddenDimSize) )
            o = tf.nn.sigmoid( self._slice(wx,3,hiddenDimSize) + self._slice(uh,3,hiddenDimSize) )
            new_ct = f * ct + i * c

            h_new = o * tf.nn.tanh(new_ct)

            h_new = tf.expand_dims(h_new,axis = 0)
            new_ct = tf.expand_dims(new_ct,axis = 0)
            h_ct = tf.concat([h_new,new_ct],axis=0)

            return h_ct

        Wx = tf.einsum("ijk,kl->ijl",day_emb,W_gru) + b_gru
        gru_results = tf.scan(fn=stepFn,elems=Wx,initializer=self.first)[:,0,:,:]

        return gru_results


    def attentionStep(self, x, att_timesteps):

        day_emb = self.day_emb[:att_timesteps]   # len × patients_num × day_dim
        rnn_h = self.gru_layer(day_emb, self.rnn_hiddendim) # len × patients_num × rnn_hiddendim

        # 逆序的LSTM
        day_emb_reverse = self.day_emb[:att_timesteps][::-1]
        rnn_h_reverse = self.gru_layer(day_emb_reverse, self.rnn_hiddendim,True)[::-1]

        # 拼接正序和逆序的LSTM輸出
        rnn_h = tf.concat([rnn_h,rnn_h_reverse],axis=2)
        # rnn_h = rnn_h_reverse

        Alpha = tf.einsum("ijk,kl->ijl", rnn_h, self.tparams["w_alpha"]) + self.tparams["b_alpha"]
        Alpha = tf.squeeze(Alpha, axis=2)
        Alpha = tf.transpose(tf.nn.softmax(tf.transpose(Alpha)))
        # Beta = tf.einsum("ijk,kl->ijl", rnn_h, self.tparams["w_beta"]) + self.tparams["b_beta"]

        # c_t = tf.reduce_mean( tf.expand_dims(Alpha,2) * self.day_emb[:att_timesteps], axis=0 )
        c_t = tf.reduce_mean( tf.expand_dims(Alpha,2) * rnn_h , axis=0 )

        h_t = tf.concat( [c_t,rnn_h[-1]],axis=1)
        print("213")
        print(h_t)

        h_t_out = tf.matmul(h_t,self.tparams["W_c"])

        return h_t_out

    def build_net(self):

        # 第一层降维
        self.day_emb = tf.einsum("ijk,kl->ijl",self.x,self.tparams["input_w"]) + self.tparams["input_b"]

        # LSTM层
        if self.keep_prob < 1.0 :
            self.day_emb = tf.nn.dropout(self.day_emb,self.keep_prob)

        self.rnn_results = tf.scan(fn=self.attentionStep,elems=self.counts,initializer=self.first_ct)
        # self.rnn_results = self.gru_layer(self.day_diag_emb,self.rnn_hiddendim,self.gapday)
        # self.rnn_results = tf.squeeze(self.rnn_results[:,0,:,:],1)

        # 输出层
        self.y_hat = tf.einsum("ijk,kl->ijl",self.rnn_results,self.tparams["output_w"]) + self.tparams["output_b"]
        self.y_hat = tf.nn.sigmoid(self.y_hat)

        # 损失函数
        loss = tf.reduce_sum( -(self.y * tf.log(self.y_hat + self.L2)  + (1.0 - self.y) * tf.log(1. - self.y_hat + self.L2)) )

        # self.loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=self.y_hat) )
        return loss


    # 对原始的seqs(patients × visits × medical code)进行处理，得到便于RNN训练的结构（maxlen × n_samples × inputDimSize）
    # 返回的length表示这个seqs中所有病人对应的visit的数目
    # 原始的medical code对应的是一组tuple
    # def padTrainMatrix(self,seqs, v2diag, inputDimSize=4216, diagDimSize=824):
    #     lengths = np.array([len(seq) for seq in seqs]).astype("int32")
    #     n_samples = len(seqs)
    #     maxlen = np.max(lengths)
    #
    #     x = np.zeros([maxlen, n_samples, inputDimSize + diagDimSize]).astype(np.float32)
    #     x_onehot = np.zeros([maxlen, n_samples, inputDimSize]).astype(np.float32)
    #     for idx, seq in enumerate(seqs):
    #         diag_id = v2diag[idx]
    #         for xvec, subseq in zip(x[:, idx, :], seq):
    #             for tuple in subseq:
    #                 xvec[tuple[0]] = 1
    #
    #             # 将第4216+i位置对应第i个疾病，将其设置为1
    #             xvec[4216 + diag_id - 1] = 1
    #
    #         # 构造onehot的表示形式作为label
    #         for xvec_onehot, subseq in zip(x_onehot[:, idx, :], seq):
    #             for tuple in subseq:
    #                 xvec_onehot[tuple[0]] = 1
    #
    #     return x, x_onehot, lengths,maxlen
    def padTrainMatrix(self,seqs):
        lengths = np.array( [ len(seq) for seq in seqs ] ).astype("int32")
        n_samples = len(seqs)
        maxlen = np.max(lengths)

        x = np.zeros([maxlen,n_samples,self.input_dim]).astype(np.float32)
        for idx,seq in enumerate(seqs):
            for xvec,subseq in zip(x[:,idx,:],seq):
                for tuple in subseq:
                    xvec[tuple[0]] = tuple[1]
        return x,lengths

    # 将patients的矩阵变为 maxlen * patients * 2,用于与Daydiag矩阵进行操作
    def padDiagsMatrix(self,v2diag,maxlen,d2trueday_batch):

        patients_num = len(v2diag)
        # v2diag-=1
        x = np.zeros(shape=[maxlen,patients_num,2],dtype=np.int32)

        for i in range(patients_num):
            for j in range(maxlen):
                x[j][i][0] = v2diag[i]
                if j >= len(d2trueday_batch[i]):
                    x[j][i][1] = 49
                else:
                    x[j][i][1] = d2trueday_batch[i][j]
        return x

    # 根据maxgap，d2trueday 生成 gapDayemb [patient × maxlen × maxgap × 2]
    def getGapDayEmb(self,gap_batch,d2trueday_batch,maxlen,v2diag):

        # for gap in gap_batch:
        #     print str( len(gap) ) + " ",
        # print("----")
        #
        # for d2trueday in d2trueday_batch:
        #     print str( len(d2trueday) ) + " ",
        # print("----")

        maxgap = 0
        for daygap in gap_batch:
            temp_max = np.max(daygap)
            maxgap = np.maximum(maxgap,temp_max)
        patients_num = len(d2trueday_batch)

        # print(gap_batch)
        #
        # print("maxlen:" + str(maxlen) )
        # print("patients_num:" + str(patients_num) )
        # print("maxgap:" + str(maxgap))

        if maxgap == 0:
            maxgap = 1
        x = np.zeros(shape=[maxlen,patients_num,maxgap,2])
        x[:,:,:,1] = 49

        for i in range(patients_num):
            for j in range(maxlen):
                for m in range(maxgap):
                    x[j][i][m][0] = v2diag[i]
                    # if j >= len(gap_batch[i]):
                    #     x[j][i][m][1] = 49
                    # else:
                    #     if gap_batch[i][j] == 0:
                    #         x[j][i][m][1] = 49
                    #     else:
                    #         pre_day = d2trueday_batch[i][j] - gap_batch[i][j] + m
                    #         x[j][i][m][1] = pre_day

        for i in range(patients_num):
            for j in range(maxlen):
                if j < len(gap_batch[i]):
                    for m in range(gap_batch[i][j]):
                        x[j][i][m][1] = d2trueday_batch[i][j] - gap_batch[i][j] + m

        # print(x)
        return x

    def startTrain(self,x=None,y=None):
        counts = np.arange(x.shape[0]) + 1
        first_ct = np.zeros([x.shape[1],self.day_dim])
        first = np.zeros([2,x.shape[1],self.rnn_hiddendim])
        loss,y_hat,y,opt = self.sess.run((self.loss,self.y_hat,self.y,self.opt),feed_dict={self.x:x,self.y:y,self.first_ct:first_ct,
                                                                                           self.counts:counts,self.first:first})

        return loss,y_hat

    def get_results(self,x=None,y=None):
        counts = np.arange(x.shape[0]) + 1
        first_ct = np.zeros([x.shape[1], self.day_dim])
        first = np.zeros([2, x.shape[1], self.rnn_hiddendim])
        loss, y_hat = self.sess.run((self.loss, self.y_hat),
                                            feed_dict={self.x: x, self.y: y,self.first_ct:first_ct,
                                                       self.counts:counts,self.first:first})
        return loss, y_hat


    def get_params(self):
        return self.tparams

