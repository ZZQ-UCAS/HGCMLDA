import csv
import os
import torch as t
import numpy as np
from math import e
import pandas as pd
from scipy import io
import torch


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)

def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)

def read_mat(path, name):
    matrix = io.loadmat(path)
    matrix = t.FloatTensor(matrix[name])
    return matrix

def read_md_data(path, validation):
    result = [{} for _ in range(validation)]
    for filename in os.listdir(path):
        data_type = filename[filename.index('_')+1:filename.index('.')-1]
        num = int(filename[filename.index('.')-1])
        result[num-1][data_type] = read_csv(os.path.join(path, filename))
    return result

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)

def Gauss_L(adj_matrix, N):
    GL = np.zeros((N, N))
    rl = N * 1. / sum(sum(adj_matrix * adj_matrix))
    for i in range(N):
        for j in range(N):
            GL[i][j] = e ** (-rl * (np.dot(adj_matrix[i, :] - adj_matrix[j, :], adj_matrix[i, :] - adj_matrix[j, :])))
    return GL

def Gauss_D(adj_matrix, M):
    GD = np.zeros((M, M))
    T = adj_matrix.transpose()
    rd = M * 1. / sum(sum(T * T))
    for i in range(M):
        for j in range(M):
            GD[i][j] = e ** (-rd * (np.dot(T[i] - T[j], T[i] - T[j])))
    return GD


def prepare_data(opt):
    dataset = {}
    dd_data  = np.loadtxt(opt.data_path +'DSS.txt', dtype=np.float64)
    dd_mat = np.array(dd_data)
    ll_data  = np.loadtxt(opt.data_path +'LFS.txt', dtype=np.float64)
    ll_mat = np.array(ll_data)
    lnc_dis_data = np.loadtxt(opt.data_path +'LDA.txt', dtype=np.float64)
    dataset['ld_p'] = t.FloatTensor(np.array(lnc_dis_data))
    dataset['ld_true'] = dataset['ld_p']
    all_zero_index = []
    all_one_index = []
    for i in range(dataset['ld_p'].size(0)):
        for j in range(dataset['ld_p'].size(1)):
            if dataset['ld_p'][i][j] < 1:
                all_zero_index.append([i, j])
            if dataset['ld_p'][i][j] >= 1:
                all_one_index.append([i, j])
    np.random.seed(0)
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)
    in_zero_index_test=np.loadtxt('independent_negative.txt',dtype=np.int16)
    in_one_index_test=np.loadtxt('independent_positive.txt',dtype=np.int16)
    in_zero_index_test=np.array(in_zero_index_test)
    in_one_index_test=np.array(in_one_index_test)
    all_zero_index=np.array(all_zero_index)
    all_one_index=np.array(all_one_index)

    df_all_one_index = pd.DataFrame(all_one_index, columns=['Row', 'Col'])
    df_in_one_index_test = pd.DataFrame(in_one_index_test, columns=['Row', 'Col'])
    cross_one_index_temp = pd.merge(df_all_one_index, df_in_one_index_test, on=['Row', 'Col'], how='left', indicator=True)
    cross_one_index_temp = cross_one_index_temp[cross_one_index_temp['_merge'] == 'left_only']
    cross_one_index_temp = cross_one_index_temp[['Row', 'Col']]
    cross_one_index=cross_one_index_temp.to_numpy()
    
    df_all_zero_index = pd.DataFrame(all_zero_index, columns=['Row', 'Col'])
    df_in_zero_index_test = pd.DataFrame(in_zero_index_test, columns=['Row', 'Col'])
    cross_zero_index_temp = pd.merge(df_all_zero_index, df_in_zero_index_test, on=['Row', 'Col'], how='left', indicator=True)
    cross_zero_index_temp = cross_zero_index_temp[cross_zero_index_temp['_merge'] == 'left_only']
    cross_zero_index_temp = cross_zero_index_temp[['Row', 'Col']]
    cross_zero_index=cross_zero_index_temp.to_numpy()

    cross_one_index=t.LongTensor(cross_one_index)
    cross_zero_index=t.LongTensor(cross_zero_index)

    in_zero_index_test=t.LongTensor(in_zero_index_test)
    in_one_index_test=t.LongTensor(in_one_index_test)

    dataset['independent'] = []
    dataset['independent'].append({'test': [in_one_index_test, in_zero_index_test]}) 

    new_zero_index = cross_zero_index.split(int(cross_zero_index.size(0) / opt.validation), dim=0)
    new_one_index = cross_one_index.split(int(cross_one_index.size(0) / opt.validation), dim=0)

    dataset['ld'] = []

    if int(cross_one_index.size(0) % opt.validation)!=0:
        for i in range(opt.validation):
            a = [s for s in range(opt.validation+1)]
            if i == opt.validation - 1:
                temp1=dataset['ld_p'].clone()
                temp2=t.cat([new_one_index[j] for j in [-2,-1]])
                temp3=in_one_index_test
                temp1[temp2[:,0],temp2[:,1]]=0    
                temp1[temp3[:,0],temp3[:,1]]=0
                del a[i:i+2]
                dataset['ld'].append({'test': [t.cat([new_one_index[j] for j in [-2,-1]]),t.cat([new_zero_index[j] for j in [-2,-1]])],
                              'train': [t.cat([new_one_index[j] for j in a]), t.cat([new_zero_index[j] for j in a])],'train_adj':temp1})
            else:
                temp1=dataset['ld_p'].clone()
                temp2=new_one_index[i]
                temp3=in_one_index_test
                temp1[temp2[:,0],temp2[:,1]]=0    
                temp1[temp3[:,0],temp3[:,1]]=0
                del a[i]
                dataset['ld'].append({'test': [new_one_index[i], new_zero_index[i]],
                              'train': [t.cat([new_one_index[j] for j in a]), t.cat([new_zero_index[j] for j in a])],'train_adj':temp1})
    else:
        for i in range(opt.validation):
            temp1=dataset['ld_p'].clone()
            temp2=new_one_index[i]
            temp3=in_one_index_test
            temp1[temp2[:,0],temp2[:,1]]=0    
            temp1[temp3[:,0],temp3[:,1]]=0
            a = [s for s in range(opt.validation)]
            del a[i]
            dataset['ld'].append({'test': [new_one_index[i], new_zero_index[i]],
                              'train': [t.cat([new_one_index[j] for j in a]), t.cat([new_zero_index[j] for j in a])],'train_adj':temp1})


    DGSM = Gauss_D(dataset['ld_true'].numpy(), dataset['ld_true'].size(1))
    LGSM = Gauss_L(dataset['ld_true'].numpy(), dataset['ld_true'].size(0))

    nd = lnc_dis_data.shape[1]
    nl = lnc_dis_data.shape[0]

    ID = np.zeros([nd, nd])

    for h1 in range(nd):
        for h2 in range(nd):
            if dd_mat[h1, h2] == 0:
                ID[h1, h2] = DGSM[h1, h2]
            else:
                ID[h1, h2] = dd_mat[h1, h2]

    IL = np.zeros([nl, nl])

    for q1 in range(nl):
        for q2 in range(nl):
            if ll_mat[q1, q2] == 0:
                IL[q1, q2] = LGSM[q1, q2]
            else:
                IL[q1, q2] = ll_mat[q1, q2]

    dataset['ID'] = t.from_numpy(ID)
    dataset['IL'] = t.from_numpy(IL)

    return dataset
