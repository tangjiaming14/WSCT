import config
import pickle
import myTools
import numpy
import scipy.sparse as ss
from scipy import linalg

save_A_path = './A.pkl'
save_L_path = './L.pkl'

with open(save_A_path,'rb') as f:       #获取相似矩阵A数据
    A_value = pickle.load(f)
    image_num = pickle.load(f)
print(A_value)
print(image_num)



