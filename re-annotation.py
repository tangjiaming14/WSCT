import myTools as mt
import pickle
from pylab import *
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten
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

#获取原始label数据
test_reader = pd.read_csv(labels_path)
label_value = test_reader.values
label_value = mt.matrix_to_1D(label_value)

#获取wlbl数据
test_reader = pd.read_csv(wlbl_path)
wlbl_value = test_reader.values
wlbl_value = mt.matrix_to_1D(wlbl_value)


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


D = ROC.rank_order_cluster(S, n)  #通过计算秩序距离获得距离矩阵
'''
disMat = scipy.spatial.distance.pdist(Wt, 'seuclidean')  #标准欧式距离，获得压缩距离数组，大小(n*(n-1)/2)

Z = sch.linkage(disMat, method ='centroid') #合并n-1次，为(n-1)*2大小的矩阵，代表每次合并的结果
#sch.dendrogram(Z)  #画出树状图
#plt.show()
#t = [8.0]
#t = [8.0, 5.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.07, 0.01]  #设定每个簇间点的距离不存在大于t的点 ward
t = [5.0,1.0,0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01] #median
for i in t:
    cluster= sch.fcluster(Z, t=i, criterion = 'distance')     #获取聚类结果，聚类数为max的值
    umqi = ROC.uMQI(cluster, S, i)
    wlbl_change = ROC.re_annotation(wlbl_value, cluster)
    acc = mt.accuracy(wlbl_change, label_value)
    print('t = ', i, 'num of clusters = ', max(cluster), 'umqi = ', umqi, 'acc = ', acc)
'''