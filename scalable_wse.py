from pylab import *
import numpy as np
import pandas as pd
import pickle
import config
import myTools as mt
import scipy.sparse as ss
from scipy import linalg
import wse
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

wlbl_path = './data/fer2013/wlbl_clu.csv'
save_A_path = './A.pkl'
save_L_path = './L.pkl'

wlbl_reader = pd.read_csv(wlbl_path)    #获取wlbl数据
wlbl_values = wlbl_reader.values
wlbl_values = mt.matrix_to_1D(wlbl_values)

with open(save_A_path,'rb') as f:       #获取相似矩阵A数据
    A_value = pickle.load(f)
    image_num = pickle.load(f)

config = config.get_config('wse', 'toy') #获取配置

outdim = mt.get_field(config, 'embedding_dim', 2)
type_lap = mt.get_field(config, 'type_laplacian', 'norm')
speedup = mt.get_field(config, 'speedup', 'fast')
optimizer = mt.get_field(config, 'optimizer', 'agd')
epoch = mt.get_field(config, 'epoch', 50)
graph_partition = mt.get_field(config, 'graph_partition', 2)
log_opt = mt.get_field(config, 'log_opt', {})
lambdas = mt.get_field(config, 'lambda', 0.3413)
eta = mt.get_field(config, 'eta', 0.1)
gamma = mt.get_field(config, 'gamma', 0.9)

assert lambdas != 0, 'lambda=0'

# Calculate degree matrix
degs = mt.sum_by_row(A_value, image_num)

a = np.ones((image_num,image_num))*0
D = ss.lil_matrix(a)                                                 #有限图的度矩阵D
for i in range(image_num):
    D[i, i] = degs[i]

L = D - A_value                                                     #计算拉普拉斯矩阵L=D-A
#print(L)
# Compute normalized Laplacian if needed
if type_lap == 'norm':
    # avoid dividing by zero
    for i in range(image_num):
        if degs[i] == 0:
            degs[i] = 2.2204e-16
    #calculate ingerse of D
    for i in range(image_num):
        D[i, i] = 1/degs[i]
    #calculate normalized Laplacian
    L = mt.mul_by_pos(L, D, image_num)                          #L对角线为1 其他位是之前的矩阵值

#存储数据
#存储L
try:
    g = open(save_L_path, 'wb')
    pickle.dump(L,g)
    g.close()
except e:
        print('存储错误', e)

