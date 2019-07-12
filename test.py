import config
import pickle
import myTools as mt
import numpy
import scipy.sparse as ss
from scipy import linalg
import pandas as pd

save_A_path = './A.pkl'
save_L_path = './L.pkl'
wlbl_path = './data/fer2013/wlbl_clu.csv'

lambdas = 0.3413
eta = 0.1

with open(save_L_path,'rb') as f:       #获取L数据
    L_value = pickle.load(f)

Wt = linalg.orth(numpy.random.random(size=(3589,2)))    #随机生成image_num*outdim的0~1的矩阵，再求他的正交基W0
V = Wt
n = numpy.size(V, 0)
wlbl_reader = pd.read_csv(wlbl_path)
wlbl_values = wlbl_reader.values
wlbl_values = mt.matrix_to_1D(wlbl_values)

wlbl_t = numpy.unique(wlbl_values)

gind = numpy.argwhere(wlbl_values == wlbl_t[0])
gind = mt.matrix_to_1D(gind)
ng = numpy.size(gind)
Cg = numpy.identity(ng) - (numpy.ones((ng,ng))/ng)
temp = numpy.matrix(V[gind, ])
temp1 = temp.T * Cg
trace = numpy.trace(numpy.dot(temp1, temp))
print(trace)
obj2 = n/( numpy.size(wlbl_t)* ng)*trace
print(obj2)

obj = numpy.trace(Wt.T*L_value*Wt) + \
      numpy.trace( numpy.matrix((Wt-V).T)* (2*L_value*V) ) + sum(mt.sum_diag((Wt-V).T*(Wt-V))) / (2*eta) + lambdas *obj2
print(obj)