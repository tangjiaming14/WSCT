import config
import pickle
import myTools as mt
import numpy
import scipy.sparse as ss
from scipy import linalg
import pandas as pd
import wse
import liteKmeans
from scipy.cluster.vq import *

save_A_path = './A.pkl'
save_L_path = './L.pkl'
wlbl_path = './data/fer2013/wlbl_clu.csv'

lambdas = 0.3413
eta = 0.1

with open(save_L_path,'rb') as f:       #获取L数据
    L_value = pickle.load(f)
k = 2

Wt = linalg.orth(numpy.random.random(size=(3589,k)))    #随机生成image_num*outdim的0~1的矩阵，再求他的正交基W0
V = Wt
n = numpy.size(V, 0)
wlbl_reader = pd.read_csv(wlbl_path)
wlbl_values = wlbl_reader.values
wlbl_values = mt.matrix_to_1D(wlbl_values)
wlbl_t = numpy.unique(wlbl_values)

code,distance = liteKmeans.Kmeans(Wt,k)
ct = {}
ct[1] = code
print(ct)