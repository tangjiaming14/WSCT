from pylab import *
import numpy as np
import scipy.sparse as ss
import math

def get_Max(a, b, n):
    c = np.ones((n,n))*0
    result = ss.lil_matrix(c)
    for i in range(n):
        for j in range(n):
            result[i, j] = max(a[i,j],b[i,j])
    return result

def get_Min(a, b, n):
    c = np.ones((n,n))*0
    result = ss.lil_matrix(c)
    for i in range(n):
        for j in range(n):
            result[i,j] = min(a[i,j], b[i,j])

    return result

def get_median(knndist, isnn ,n):
    term = []
    for i in range(n):
        for j in range(n):
            if isnn[j,i] == 1:
                term.append(knndist[j,i])
    l = len(term) #数出列表中有几个元素，将个数放到l里
    term.sort()
    if l == 0:
        m = 0
    elif l%2 == 0:
        m = (term[int(l/2) - 1] + term[int(l/2)]) / 2
    else:
        m = term[int((l-1)/2)]

    print(m)
    return m

def sim_gaussian(knndist, sigman,n):
    a = np.ones((n,n))*0
    result = ss.lil_matrix(a)
    for i in range(n):
        for j in range(n):
            if(knndist[i,j] != 0):
                result[i,j] = math.exp(-(knndist[i,j]*knndist[i,j]) / (2 * (sigman*sigman)))

    return result

def get_randomInt(n, m, x):
    a = []
    i = 0
    while i < x:
        tem = np.random.randint(n, m)
        if tem not in a:
            a.append(tem)
            i += 1
    return a
def perturb_labels(labels, psnr):     #污染真实数据得到弱注释数据
    label_set = np.unique(labels)
    perturbed_labels = labels

    for i in label_set:
        label = i
        idx = np.argwhere(labels == label)
        n = len(idx)

        sample_set = [val for val in label_set if val != label]   #获取label_set中不是label的标签（除label外的标签）
        num_perturb = math.floor(psnr*n)     #向下取整

        pos = np.random.randint(0, len(sample_set),(num_perturb,1))
        perturb_class_ids = np.ones((num_perturb,1))*0
        for j in range(len(pos)):                           #根据随机生成的坐标值，生成随机的标签值
            perturb_class_ids[j, 0] = sample_set[pos[j, 0]]

        perturb_idx = get_randomInt(0, n, num_perturb)

        for j in range(len(perturb_idx)):
            perturbed_labels[idx[perturb_idx[j]]] = perturb_class_ids[j, 0]
    return perturbed_labels

def label_to_couple(labels):    #将标签为3，5，6的变为1，其他为0
    result_label = np.ones((size(labels), 1))*0
    for i in range(size(labels)):
        if labels[i, 0] == 3 or labels[i, 0] == 5 or labels[i, 0] == 6:
            result_label[i, 0] = 1
    return result_label

def get_field(config, key, default):   #获取config的值
    if key in config:
        return config[key]
    else:
        return default

def sum_by_row(A, n):         #计算每一行的数值和
    a = np.ones((n,1))*0
    for i in range(n):
        row_sum = 0
        for j in range(n):
            row_sum = row_sum + A[i, j]
        a[i,0] = row_sum
    return a

def sum_by_column(A):         #计算每一列的数值和
    m = np.size(A, 0)
    n = np.size(A, 1)
    a = np.ones((1,n))*0
    for i in range(n):
        column_sum = 0
        for j in range(m):
            column_sum = column_sum + A[j, i]
        a[0,i] = column_sum
    return a

def mul_by_pos(A, B, n):         # 将AB对角线的值相乘，其余位是A的值
    a = np.ones((n,n))*0
    result = ss.lil_matrix(a)
    for i in range(n):
        for j in range(n):
            if i == j:
                result[i, j] = A[i, j]*B[i, j]
            else:
                result[i, j] = A[i ,j]
    return result

def matrix_to_1D(A):   #将2维矩阵变为1维，一列一列添加
    a = []
    for i in range(np.size(A, 1)):
        for j in range(np.size(A, 0)):
            a.append(A[j][i])

    return a

def sum_diag(A):           #计算矩阵对角线的和
    m = np.size(A, 0)
    n = np.size(A, 1)
    sum = 0
    for i in range(m):
        for j in range(n):
            if i == j:
                sum = sum + A[i, j]

    return sum

def change_to_matrix(A):     #将奇奇怪怪的数据类型都变为二维矩阵
    m = np.size(A, 0)
    n = np.size(A, 1)
    a = a = np.ones((m,n))
    for i in range(m):
        for j in range(n):
            a[i, j] = A[i, j]
    return a

def plot_sample(feat, label):    #与matlab相似
    class_ids = unique(label)
    for i in range(size(class_ids)):
        class_id = class_ids[i]
        class_ind = where(label==class_id)[0]
        xx = feat[class_ind,0]
        yy = feat[class_ind,1]
        if mean(yy)<-0.01:
            for j in range(size(xx)):
                plot(xx[j], yy[j], 'y.')
        else:
            for j in range(len(xx)):
                plot(xx[j], yy[j], 'b.')
        axis('off')

def plot_sample2(feat, label):        #多个颜色
    class_ids = unique(label)
    color = ['y.', 'b.', 'r.', 'g.', 'm.', 'c.', 'k.']
    for i in range(size(class_ids)):
        class_id = class_ids[i]
        class_ind = where(label==class_id)[0]
        xx = feat[class_ind,0]
        yy = feat[class_ind,1]
        for j in range(size(xx)):
            plot(xx[j], yy[j], color[i])
        axis('off')

def accuracy(A, B):
    num = len(A)
    count = 0
    for i in range(num):
        if A[i] == B[i]:
            count +=1
    acc = count/num
    return acc
