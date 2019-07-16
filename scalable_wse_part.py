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

with open(save_L_path,'rb') as f:       #获取L数据
    L = pickle.load(f)

# Init
W0 = linalg.orth(np.random.random(size=(image_num,outdim)))    #随机生成image_num*outdim的0~1的矩阵，再求他的正交基W0
Wt = W0
V = Wt

if optimizer == 'sgd':
    n = image_num


elif optimizer == 'agd':
    [CtA, Wt, V, obj, obj1, obj2] = wse.wse(L, Wt, V, wlbl_values, eta, gamma, lambdas, speedup, optimizer, log_opt)



# display

num = len(CtA) - 1
figure()
ndx = where(CtA[num] == 0)[0]
plot(Wt[ndx, 0], Wt[ndx, 1], '*')
ndx = where(CtA[num] == 1)[0]
plot(Wt[ndx, 0], Wt[ndx, 1], 'r.')
title(u'2维数据点聚类', fontproperties=font)
axis('off')
show()