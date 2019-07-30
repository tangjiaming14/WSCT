 # -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from keras.models import load_model
#import pickle
import h5py
import scipy.sparse as ss
import myTools as mt
import pickle

np.seterr(divide='ignore',invalid='ignore')

IMAGE_SIZE = 160 # 指定图像大小  use Facenet

def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0,0,0,0)

    # 获取图片尺寸
    h, w, _ = image.shape

    # 对于长宽不等的图片，找到最长的一边
    longest_edge = max(h,w)

    # 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass # pass是空语句，是为了保持程序结构的完整性。pass不做任何事情，一般用做占位语句。

    # RGB颜色
    BLACK = [0,0,0]
    # 给图片增加padding，使图片长、宽相等
    # top, bottom, left, right分别是各个边界的宽度，cv2.BORDER_CONSTANT是一种border type，表示用相同的颜色填充
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度
    return cv2.resize(constant, (height, width))

def img_to_encoding(images, model):
    # 这里image的格式就是opencv读入后的格式
    images = images[...,::-1] # Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode. 这里的操作实际是对channel这一dim进行reverse，从BGR转换为RGB
    images = np.around(images/255.0, decimals=12) # np.around是四舍五入，其中decimals是保留的小数位数,这里进行了归一化
    # https://stackoverflow.com/questions/44972565/what-is-the-difference-between-the-predict-and-predict-on-batch-methods-of-a-ker
    if images.shape[0] > 1:
        embedding = model.predict(images, batch_size = 256) # predict是对多个batch进行预测，这里的256是尝试后得出的内存能承受的最大值
    else:
        embedding = model.predict_on_batch(images) # predict_on_batch是对单个batch进行预测
    # 报错，operands could not be broadcast together with shapes (2249,128) (2249,)，因此要加上keepdims = True
    embedding = embedding / np.linalg.norm(embedding, axis = 1, keepdims = True) # 注意这个项目里用的keras实现的facenet模型没有l2_norm，因此要在这里加上

    return embedding


if __name__ == "__main__":
    save_A_path = './A.pkl'
    save_feat_path = './result/feat.pkl'
    image_path = './data/fer2013/cluTest/'
    facenet = load_model('./model/facenet_keras.h5')
    #facenet.summary()

    images = []
    person_pics = os.listdir(image_path)
    for file in person_pics:
        imag = cv2.imread(os.path.join(image_path, file))
        if imag is None:
            pass
        else:
            imag = resize_image(imag, IMAGE_SIZE, IMAGE_SIZE)
            images.append(imag)

    images = np.array(images)
    #print(np.size(images))

    img_embedding = img_to_encoding(images, facenet)
    n = len(img_embedding)
    #print(n)

    nn_opt = 'pdist'

if nn_opt == 'pdist':
    print('create A by pdist')
    #Create directed neighbor graph

    # compute distance matrix
    S = np.array([[ np.sqrt(sum((img_embedding[i]-img_embedding[j])**2))       #3589*3589
              for i in range(n) ] for j in range(n)], 'f')

    graph_type = 'mutual_knn'
    k = 300  #构建图的最近邻居数
    isnn = np.ones((n,n))*0

    for irow in range(n):              #将距离从小到大排序，选取最小的k个索引在isnn中赋值为1
        idx = S[irow,:].argsort()
        isnn[irow, idx[1:k+1]] = 1

    a = np.ones((n,n))*0
    knndist = ss.lil_matrix(a)     #生成稀疏矩阵

    for i in range(n):                   #knndist(isbn) = S(isbn)
        for j in range(n):
            if isnn[i,j]==1:
                knndist[i,j] = S[i,j]

    if graph_type == 'mutual_knn':
        knndist = mt.get_Min(knndist, knndist.T, n)
    elif graph_type == 'knn':
        knndist = mt.get_Max(knndist, knndist.T, n)

    #print(knndist.data)
    #print(knndist.rows)
    #print(knndist.nnz)
    #Gaussian parameter
    sigma = mt.get_median(knndist, isnn, n)

    A = mt.sim_gaussian(knndist, sigma, n)
    print(A.data)
    print(A.rows)
    print(A.nnz)
    try:
        f = open(save_A_path, 'wb')
        pickle.dump(A,f)
        pickle.dump(n,f)
        f.close()
        g = open(save_feat_path, 'wb')
        pickle.dump(img_embedding,g)
        g.close()
    except:
        print('save error')


elif nn_opt == 'flann':
    print('create A by flann')