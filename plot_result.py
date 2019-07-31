import myTools as mt
import pickle
from pylab import *
import pandas as pd
from PCV.tools import imtools, pca
from PIL import Image, ImageDraw
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

wlbl_path = './data/fer2013/wlbl_clu.csv'
savr_result_path = './result/result.pkl'
#csv_path = './data/fer2013/test.csv'
save_feat_path = './result/feat.pkl'
labels_path = './data/fer2013/labels_clu.csv'

#获取wse结果
with open(savr_result_path,'rb') as f:
    CtA = pickle.load(f)
    obj = pickle.load(f)
    Wt = pickle.load(f)

#获取原始特征数据
with open(save_feat_path,'rb') as f:
    projected = pickle.load(f)
n = len(projected)

#获取原始label数据
test_reader = pd.read_csv(labels_path)     #读取emotion的数据
label_value = test_reader.values
label_value = mt.matrix_to_1D(label_value)

iters = len(CtA) #获取迭代次数

#获取wlbl数据
wlbl_reader = pd.read_csv(wlbl_path)
wlbl_values = wlbl_reader.values
wlbl_values = mt.matrix_to_1D(wlbl_values)

class_ids = unique(wlbl_values)
pro = mt.change_to_matrix(projected)
'''
figure()
gray()
subplot(131)
mt.plot_sample(pro, wlbl_values)
subplot(132)
mt.plot_sample(pro, CtA[iters-1])
subplot(133)
mt.plot_sample(pro, label_value)
show()
'''
ol = iters-1
print(mt.accuracy(wlbl_values, CtA[ol]))
print(mt.accuracy(wlbl_values, label_value))
print(mt.accuracy(label_value, CtA[ol]))