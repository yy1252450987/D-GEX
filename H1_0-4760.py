#!/usr/bin/env python

import sys
import os
import pickle as pkl
import time
import random

import numpy as np
import theano.tensor as T
import theano
import pylearn2.train
# 训练 类
import pylearn2.models.mlp as p2_md_mlp
# 多层感知机 类
import pylearn2.datasets.dense_design_matrix as p2_dt_dd
# 稠密矩阵 数据 类
import pylearn2.training_algorithms.sgd as p2_alg_sgd
# 训练算法 类
import pylearn2.training_algorithms.learning_rule as p2_alg_lr
# 学习规则 类
import pylearn2.costs.mlp.dropout as p2_ct_mlp_dropout
# dropout 类
import pylearn2.termination_criteria as p2_termcri
# 终止标准 类
from numpy import dtype
# dtype 数据类

def main():
    base_name = sys.argv[1] #文件名前缀
    n_epoch = int(sys.argv[2]) # epoch次数
    n_hidden = int(sys.argv[3]) # 隐含层节点数
    include_rate = float(sys.argv[4]) # 包含率（1-dropout）

    in_size = 943 # 输入层节点数目
    out_size = 4760  #输出层节点数
    b_size = 200 #batch的大小
    l_rate = 5e-4 #学习速率
    l_rate_min = 1e-5 #学习速率最小值
    decay_factor = 0.9 #
    lr_scale = 3.0 #
    momentum = 0.5 #摄动因子
    init_vals = np.sqrt(6.0/(np.array([in_size, n_hidden])+np.array([n_hidden, out_size])))
    
    print 'loading data...'
    #读取数据Train,Validation,Test
    X_tr = np.load('bgedv2_X_tr_float64.npy')
    Y_tr = np.load('bgedv2_Y_tr_0-4760_float64.npy')
    Y_tr_target = np.array(Y_tr)
    X_va = np.load('bgedv2_X_va_float64.npy')
    Y_va = np.load('bgedv2_Y_va_0-4760_float64.npy')
    Y_va_target = np.array(Y_va)
    X_te = np.load('bgedv2_X_te_float64.npy')
    Y_te = np.load('bgedv2_Y_te_0-4760_float64.npy')
    Y_te_target = np.array(Y_te)

    X_1000G = np.load('1000G_X_float64.npy')
    Y_1000G = np.load('1000G_Y_0-4760_float64.npy')
    Y_1000G_target = np.array(Y_1000G)
    X_GTEx = np.load('GTEx_X_float64.npy')
    Y_GTEx = np.load('GTEx_Y_0-4760_float64.npy')
    Y_GTEx_target = np.array(Y_GTEx)

    #随机化
    random.seed(0)
    #随机抽取5000样本进行训练
    monitor_idx_tr = random.sample(range(88807), 5000)
    #将数据X,Y整合成DensenMatrix类型
    data_tr = p2_dt_dd.DenseDesignMatrix(X=X_tr.astype('float32'), y=Y_tr.astype('float32'))
    #取出X中对应5000样本进行训练
    X_tr_monitor, Y_tr_monitor_target = X_tr[monitor_idx_tr, :], Y_tr_target[monitor_idx_tr, :]
    #设置多层感知机的隐含层计算方式
    h1_layer = p2_md_mlp.Tanh(layer_name='h1', dim=n_hidden, irange=init_vals[0], W_lr_scale=1.0, b_lr_scale=1.0)
    #设置多层感知机的输出层计算方式
    o_layer = p2_md_mlp.Linear(layer_name='y', dim=out_size, irange=0.0001, W_lr_scale=lr_scale, b_lr_scale=1.0)
    #设置好模型 
    model = p2_md_mlp.MLP(nvis=in_size, layers=[h1_layer, o_layer], seed=1)
    #设置dropout比例
    dropout_cost = p2_ct_mlp_dropout.Dropout(input_include_probs={'h1':1.0, 'y':include_rate}, 
                                             input_scales={'h1':1.0, 
                                                           'y':np.float32(1.0/include_rate)})
    #设置训练算法（batch大小，学习速率，学习规则，终止条件，dropout比例）
    algorithm = p2_alg_sgd.SGD(batch_size=b_size, learning_rate=l_rate, 
                               learning_rule = p2_alg_lr.Momentum(momentum),
                               termination_criterion=p2_termcri.EpochCounter(max_epochs=1000),
                               cost=dropout_cost)
    #设置训练类（数据集，训练模型，训练算法）
    train = pylearn2.train.Train(dataset=data_tr, model=model, algorithm=algorithm)
    train.setup()

    x = T.matrix()
    y = model.fprop(x) #训练好的模型对X的预测值
    f = theano.function([x], y) 

    MAE_va_old = 10.0
    MAE_va_best = 10.0
    MAE_tr_old = 10.0
    MAE_te_old = 10.0
    MAE_1000G_old = 10.0
    MAE_1000G_best = 10.0
    MAE_GTEx_old = 10.0

    outlog = open(base_name + '.log', 'w')
    log_str = '\t'.join(map(str, ['epoch', 'MAE_va', 'MAE_va_change', 'MAE_te', 'MAE_te_change', 
                              'MAE_1000G', 'MAE_1000G_change', 'MAE_GTEx', 'MAE_GTEx_change',
                              'MAE_tr', 'MAE_tr_change', 'learing_rate', 'time(sec)']))
    print log_str
    outlog.write(log_str + '\n')
    sys.stdout.flush() #刷新缓冲区

    for epoch in range(0, n_epoch):
        t_old = time.time() #开始时间
        train.algorithm.train(train.dataset)#训练
        #计算不同数据集预测值
        Y_va_hat = f(X_va.astype('float32')).astype('float64')
        Y_te_hat = f(X_te.astype('float32')).astype('float64')
        Y_tr_hat_monitor = f(X_tr_monitor.astype('float32')).astype('float64')
        Y_1000G_hat = f(X_1000G.astype('float32')).astype('float64')
        Y_GTEx_hat = f(X_GTEx.astype('float32')).astype('float64')
        #计算预测值与真实值的MAE
        MAE_va = np.abs(Y_va_target - Y_va_hat).mean()
        MAE_te = np.abs(Y_te_target - Y_te_hat).mean()
        MAE_tr = np.abs(Y_tr_monitor_target - Y_tr_hat_monitor).mean()
        MAE_1000G = np.abs(Y_1000G_target - Y_1000G_hat).mean()
        MAE_GTEx = np.abs(Y_GTEx_target - Y_GTEx_hat).mean()
        #计算迭代误差
        MAE_va_change = (MAE_va - MAE_va_old)/MAE_va_old
        MAE_te_change = (MAE_te - MAE_te_old)/MAE_te_old
        MAE_tr_change = (MAE_tr - MAE_tr_old)/MAE_tr_old
        MAE_1000G_change = (MAE_1000G - MAE_1000G_old)/MAE_1000G_old
        MAE_GTEx_change = (MAE_GTEx - MAE_GTEx_old)/MAE_GTEx_old
        
        #更新MAE
        MAE_va_old = MAE_va
        MAE_te_old = MAE_te
        MAE_tr_old = MAE_tr
        MAE_1000G_old = MAE_1000G
        MAE_GTEx_old = MAE_GTEx

        
        t_new = time.time() #终止时间
        l_rate = train.algorithm.learning_rate.get_value()
        log_str = '\t'.join(map(str, [epoch+1, '%.6f'%MAE_va, '%.6f'%MAE_va_change, '%.6f'%MAE_te, '%.6f'%MAE_te_change,
                                  '%.6f'%MAE_1000G, '%.6f'%MAE_1000G_change, '%.6f'%MAE_GTEx, '%.6f'%MAE_GTEx_change,
                                  '%.6f'%MAE_tr, '%.6f'%MAE_tr_change, '%.5f'%l_rate, int(t_new-t_old)]))
        print log_str
        outlog.write(log_str + '\n')
        sys.stdout.flush()
        
        if MAE_tr_change > 0: #如果误差增大，减小学习速率
            l_rate = l_rate*decay_factor
        if l_rate < l_rate_min: #学习速率最小为l_rate_min
            l_rate = l_rate_min

        train.algorithm.learning_rate.set_value(np.float32(l_rate)) #更改训练类的学习速率参数
        #更新Validation误差值
        if MAE_va < MAE_va_best:
            MAE_va_best = MAE_va
            outmodel = open(base_name + '_bestva_model.pkl', 'wb')
            pkl.dump(model, outmodel)
            outmodel.close()    
            np.save(base_name + '_bestva_Y_te_hat.npy', Y_te_hat)
            np.save(base_name + '_bestva_Y_va_hat.npy', Y_va_hat)
        #更新1000G误差值
        if MAE_1000G < MAE_1000G_best:
            MAE_1000G_best = MAE_1000G
            outmodel = open(base_name + '_best1000G_model.pkl', 'wb')
            pkl.dump(model, outmodel)
            outmodel.close()    
            np.save(base_name + '_best1000G_Y_1000G_hat.npy', Y_1000G_hat)
            np.save(base_name + '_best1000G_Y_GTEx_hat.npy', Y_GTEx_hat)

    print 'MAE_va_best : %.6f' % (MAE_va_best)
    print 'MAE_1000G_best : %.6f' % (MAE_1000G_best)
    outlog.write('MAE_va_best : %.6f' % (MAE_va_best) + '\n')
    outlog.write('MAE_1000G_best : %.6f' % (MAE_1000G_best) + '\n')
    outlog.close()

if __name__ == '__main__':
    main()








    

