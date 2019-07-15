import config
import pickle
import myTools as mt
import numpy
import scipy.sparse as ss
from scipy import linalg
import pandas as pd
import wse

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

[objj, objj2, objj3] = wse.obj_wse(lambdas, Wt, L_value, wlbl_values)
print(objj)

obj = wse.obj_QL(Wt, V, lambdas, L_value, eta, wlbl_values)
print(obj)
#print(1)

if objj > obj:
    print('false')