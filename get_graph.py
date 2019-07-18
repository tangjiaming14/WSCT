 # -*- coding: utf-8 -*-
from PCV.tools import imtools, pca
from PIL import Image, ImageDraw
from pylab import *
from scipy.cluster.vq import *
import scipy.sparse as ss
import myTools as mt
import numpy as np
import pickle

np.seterr(divide='ignore',invalid='ignore')

save_A_path = './A.pkl'
save_feat_path = './result/feat.pkl'

imlist = imtools.get_imlist('./data/fer2013/cluTest/')
imnbr = len(imlist)

# Load images, run PCA.
immatrix = array([array(Image.open(im)).flatten() for im in imlist], 'f')
V, S, immean = pca.pca(immatrix)

# Project on 2 PCs.
#projected = array([dot(V[[0, 1]], immatrix[i] - immean) for i in range(imnbr)])  # P131 Fig6-3左图
#projected = array([dot(V[[1, 2]], immatrix[i] - immean) for i in range(imnbr)])  # P131 Fig6-3右图
projected = array([dot(V[:10],immatrix[i]-immean) for i in range(imnbr)])    #463*40 = n*40
n = len(projected)

nn_opt = 'pdist'

if nn_opt == 'pdist':
    print('create A by pdist')
    #Create directed neighbor graph

    # compute distance matrix
    S = array([[ sqrt(sum((projected[i]-projected[j])**2))       #3589*3589
              for i in range(n) ] for j in range(n)], 'f')

    graph_type = 'mutual_knn'
    k = 100   #构建图的最近邻居数
    isnn = np.ones((n,n))*0

    for irow in range(n):              #将距离从小到大排序，选取最小的k个索引在isnn中赋值为1
        idx = S[irow,:].argsort()
        isnn[irow, idx[1:k+1]] = 1

    #print('yes')

    a = np.ones((n,n))*0
    knndist = ss.lil_matrix(a)     #生成稀疏矩阵
    #print(knndist.data)
    #print(knndist.rows)
    #print(knndist.nnz)

    for i in range(n):                   #knndist(isbn) = S(isbn)
        for j in range(n):
            if isnn[i,j]==1:
                knndist[i,j] = S[i,j]

    if graph_type == 'mutual_knn':
        knndist = mt.get_Min(knndist, knndist.T, n)
    elif graph_type == 'knn':
        knndist = mt.get_Max(knndist, knndist.T, n)

    #print(knndist.data)
    #print(knndist.rows)
    #print(knndist.nnz)
    #Gaussian parameter
    sigma = mt.get_median(knndist, isnn, n)
    print(sigma)

    A = mt.sim_gaussian(knndist, sigma, n)
    print(A.data)
    print(A.rows)
    print(A.nnz)
    try:
        f = open(save_A_path, 'wb')
        pickle.dump(A,f)
        pickle.dump(n,f)
        f.close()
        g = open(save_feat_path, 'wb')
        pickle.dump(projected,g)
        g.close()
    except e:
        print('存储错误', e)


elif nn_opt == 'flann':
    print('create A by flann')