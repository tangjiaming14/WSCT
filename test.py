import config
import pickle
import myTools as mt
import numpy
from pylab import *
import scipy.sparse as ss
from scipy import linalg
import pandas as pd
import wse
from PCV.tools import imtools, pca
from PIL import Image, ImageDraw
import liteKmeans
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
save_A_path = './A.pkl'
save_L_path = './L.pkl'
wlbl_path = './data/fer2013/wlbl_clu.csv'
savr_result_path = './result/result.pkl'
csv_path = './data/fer2013/test.csv'
labels_path = './data/fer2013/labels_clu.csv'

lambdas = 0.3413
eta = 0.1
speedup = 'fast'
optimizer = 'agd'
log_opt = {}
'''
#获取wse结果
with open(savr_result_path,'rb') as f:
    CtA = pickle.load(f)
    obj = pickle.load(f)

test_reader = pd.read_csv(csv_path,usecols=['emotion'])     #读取emotion的数据
label_value = test_reader.values

iters = len(CtA)
imlist = imtools.get_imlist('./data/fer2013/cluTest/')
imnbr = len(imlist)
# Load images, run PCA.
immatrix = array([array(Image.open(im)).flatten() for im in imlist], 'f')
V, S, immean = pca.pca(immatrix)
# Project on 2 PCs.
#projected = array([dot(V[[0, 1]], immatrix[i] - immean) for i in range(imnbr)])  # P131 Fig6-3左图
#projected = array([dot(V[[1, 2]], immatrix[i] - immean) for i in range(imnbr)])  # P131 Fig6-3右图
projected = array([dot(V[:2],immatrix[i]-immean) for i in range(imnbr)])    #463*40 = n*40
n = len(projected)
'''
wlbl_reader = pd.read_csv(wlbl_path)    #获取wlbl数据
wlbl_values = wlbl_reader.values
wlbl_values = mt.matrix_to_1D(wlbl_values)

labels_reader = pd.read_csv(labels_path)    #获取wlbl数据
labels_values = labels_reader.values
labels_values = mt.matrix_to_1D(labels_values)
'''
class_ids = unique(wlbl_values)
pro = mt.change_to_matrix(projected)
'''
print(mt.accuracy(wlbl_values, labels_values))

# Init
W0 = linalg.orth(np.random.random(size=(3589,2)))    #随机生成image_num*outdim的0~1的矩阵，再求他的正交基W0
Wt = copy(W0)
V = copy(Wt)
Wt[0,0] = 1

print(1)