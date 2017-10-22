#!/usr/bin/env python
    
import sys
    
import numpy as np
from sklearn.cluster import MiniBatchKMeans

BGEDV2_DATA = 'bgedv2_float64.npy'

def main():
    data = np.load(BGEDV2_DATA)
    X = data.transpose()
    X_rand = X[np.random.permutation(129158), :]
    #np.random.permutation(129158), 生成随机数1-129158对应bgedv的样本数
    km = MiniBatchKMeans(n_clusters=100, max_iter=10, batch_size=1000, verbose=1, compute_labels=False)
    #km对象 100聚类数，最大迭代次数10， batch大小1000， 
    km.fit(X_rand)
    label = km.predict(X)
    #生成聚类后的每一个样本所对应的类簇序号
    outfile = open('bgedv2_kmeans_100_label.txt', 'w')
    
    for l in label:
        outfile.write(str(l) + '\n')
    
    outfile.close()
    

    
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
