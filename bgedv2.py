#!/usr/bin/env python

import sys

import numpy as np
import cmap.io.gct as gct

BGEDV2_GCTX = 'bgedv2_QNORM.gctx'
LM_ID = 'map_lm.txt'
TG_ID = 'map_tg.txt'
SP_IDX_NODUP = 'bgedv2_idx_nodup_K100_D1.0.txt'

def main():
    infile = open(SP_IDX_NODUP)
    samples_idx_nodup = [] #筛选后的样本序号
    for line in infile:
        samples_idx_nodup.append(int(line.strip('\n')))
    
    infile.close()
    samples_idx_nodup = np.random.permutation(samples_idx_nodup).tolist()
    #对筛选后的样本序号列表进行随机化
    
    
    lm_id = [] 
    lm_probs_dict = {}
    infile = open(LM_ID)
    for line in infile:
        ID, probs = line.strip('\n').split('\t')[1:]
        lm_id.append(ID)
        lm_probs_dict[ID] = probs.split(',')
    #生成{ENSG00000079739:[201968_s_at]}字典
    infile.close()
    lm_id = np.random.permutation(lm_id).tolist()
    # 随机化landmark基因的ID号
    tg_id = [] 
    tg_probs_dict = {}
    infile = open(TG_ID)
    for line in infile:
        ID, probs = line.strip('\n').split('\t')[1:]
        tg_id.append(ID)
        tg_probs_dict[ID] = probs.split(',')
    #生成{ENSG00000079739:[201968_s_at]}字典
    infile.close()
    tg_id = np.random.permutation(tg_id).tolist()
    # 随机化target基因的ID号
    bgedv2_gctobj = gct.GCT(BGEDV2_GCTX)
    bgedv2_gctobj.read()
    bgedv2_genes = bgedv2_gctobj.get_rids()
    bgedv2_samples = bgedv2_gctobj.get_cids()# 获取未筛选的样本名称
    samples_idx = samples_idx_nodup
    samples_id = np.array(bgedv2_samples)[samples_idx] #获取筛选后的样本名称
  
    outfile = open('bgedv2_GTEx_1000G_sp.txt', 'w')
    for i in range(0, len(samples_id)):
        outfile.write(str(samples_idx[i]) + '\t' + samples_id[i] + '\n')
    
    outfile.close()
    #bgedv2_GTEx_1000G_sp.txt： 
    #156	CPC006_U937_6H:BRD-K56343971-001-02-3:10
    #228	BRAF001_HEK293T_24H:BRD-U73308409-000-01-9:0.15625
    #43	CPC006_SKM1_6H:BRD-U88459701-000-01-8:10
    #63	CPC020_MCF7_6H:BRD-A82307304-001-01-8:10

    data_lm = np.zeros((len(lm_id), len(samples_id)), dtype='float64')
    outfile = open('bgedv2_GTEx_1000G_lm.txt', 'w')
    for i in range(len(lm_id)):
        probs_id = lm_probs_dict[lm_id[i]]
        probs_idx = map(bgedv2_genes.index, probs_id)
        probs_data = bgedv2_gctobj.matrix[np.ix_(probs_idx, samples_idx)].astype('float64')
        probs_mean = probs_data.mean(axis=0)
        data_lm[i, :] = probs_mean
        outfile.write(lm_id[i] + '\t' + ','.join(map(str, probs_idx)) + '\t' + ','.join(map(str, probs_id)) + '\n')
    
    outfile.close()

    data_tg = np.zeros((len(tg_id), len(samples_id)), dtype='float64')
    outfile = open('bgedv2_GTEx_1000G_tg.txt', 'w')
    for i in range(len(tg_id)):
        probs_id = tg_probs_dict[tg_id[i]]
        probs_idx = map(bgedv2_genes.index, probs_id)
        probs_data = bgedv2_gctobj.matrix[np.ix_(probs_idx, samples_idx)].astype('float64')
        probs_mean = probs_data.mean(axis=0)
        data_tg[i, :] = probs_mean
        outfile.write(tg_id[i] + '\t' + ','.join(map(str, probs_idx)) + '\t' + ','.join(map(str, probs_id)) + '\n')
    
    outfile.close()
    
    data = np.vstack((data_lm, data_tg))
    
    np.save('bgedv2_GTEx_1000G_float64.npy', data)
    
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
