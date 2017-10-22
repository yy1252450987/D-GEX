#!/usr/bin/env python
    
import sys
    
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

BGEDV2_DATA = 'bgedv2_float64.npy'
LABEL = 'bgedv2_kmeans_100_label.txt'
K = 100 #类簇总数 默认100
D_THRED = 1.0

def keep(pd_k, idx_k):
    n_k = pd_k.shape[0]
    I, J = np.where(pd_k < D_THRED)
    set_k = set(range(0, n_k))
    
    for i in range(0, I.size):
        idx_i = I[i]
        idx_j = J[i]
        if idx_i >= idx_j:
            continue
        
        if idx_j in set_k:
            set_k.remove(idx_j)
        
    return idx_k[list(set_k)]
    

def main():
    data = np.load(BGEDV2_DATA)
    X = data.transpose()
    
    inlabel = open(LABEL) #读取类簇指示文件
    label = []
    for line in inlabel:
        label.append(int(line.strip('\n')))
    
    label = np.array(label) #对应的类簇序号
    inlabel.close()
    
    idx_keep = []
    for k in range(0, K):
        print k
        sys.stdout.flush() #缓冲输出
        idx_k = np.where(label == k)[0] #找到属于k类簇的样本序号
        X_k = X[idx_k, :] 
        pd_k = pairwise_distances(X_k, metric='euclidean', n_jobs=10) #类簇K内的所有样本对应的相似性系数
        idx_k_keep = keep(pd_k, idx_k) #剔除哪些相似性系数大于阈值的样本
        idx_keep.extend(idx_k_keep.tolist()) #添加到整个list中
        
    idx_keep = np.sort(np.array(idx_keep)).astype('int') #排序
    
    outfile = open('bgedv2_idx_nodup_K100_D1.0.txt', 'w')
    for idx in idx_keep:
        outfile.write(str(idx) + '\n')
    #输出剔除完相似性较低的样本数
    outfile.close()

    

    
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
