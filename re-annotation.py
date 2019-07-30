import myTools as mt
import pickle
from pylab import *
import pandas as pd
import ROC

wlbl_path = './data/fer2013/wlbl_clu.csv'
save_result_path = './result/result.pkl'
save_feat_path = './result/feat.pkl'
labels_path = './data/fer2013/labels_clu.csv'
S_result_path = './result/S_result.pkl'

#获取wse结果
with open(save_result_path,'rb') as f:
    CtA = pickle.load(f)
    obj = pickle.load(f)
    Wt = pickle.load(f)
n = size(Wt, 0)
'''
#获取原始特征数据
with open(save_feat_path,'rb') as f:
    projected = pickle.load(f)

#获取原始label数据
test_reader = pd.read_csv(labels_path)
label_value = test_reader.values
label_value = mt.matrix_to_1D(label_value)
'''

'''
 # compute distance matrix
S = array([[ sqrt(sum((Wt[i]-Wt[j])**2))       #3589*3589
          for i in range(n) ] for j in range(n)], 'f')

try:
    g = open(S_result_path, 'wb')            #存储wt生成的距离矩阵S的数据，Wt改变后需要重新计算
    pickle.dump(S,g)
    g.close()
except e:
        print('存储错误', e)
'''
#获取距离矩阵S的结果
with open(S_result_path,'rb') as f:
    S = pickle.load(f)

D = ROC.rank_order_cluster(S, n)