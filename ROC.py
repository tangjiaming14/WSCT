from pylab import *
import myTools as mt

ROC_result_path = './result/ROC.pkl'

def rank_order_cluster(S, n):

    RO = copy(S)
    for irow in range(n):              #将距离从小到大排序，计算出RO距离矩阵
        idx = S[irow,:].argsort().tolist()
        for jclomun in range(n):
            temp = idx.index(jclomun)
            RO[irow, jclomun] = temp/10000
        print('part1 finish row', irow, 'finished',irow,'/', n)
    print('finish part1')

    ROC = copy(RO)
    for i in range(n):              #RO矩阵改变成论文中的形式
        for j in range(n):
            k = 0
            temp = 0
            while k<n and RO[i, k] != j/10000:
                tt = RO[j, :].tolist().index(RO[i, k])
                temp += tt
                k +=1
            ROC[i, j] = temp
            print('part2 finish row', i,'finish clomun', j , 'finished',i,'/', n)

    print('finish part2')
    for i in range(n):             #计算d(a, b),存入RO
        for j in range(n):
            RO[i, j] = (ROC[i,j]+ROC[j,i])/(min(ROC[i,j], ROC[j,i]))

    print('finish part3, try save')
    try:
        g = open(ROC_result_path, 'wb')            #存储wt生成的距离矩阵S的数据，Wt改变后需要重新计算
        pickle.dump(RO,g)
        g.close()
    except e:
        print('存储错误', e)
    return RO

def uMQI(cluster, S, t):

    #初始化数据
    intraC = 0  #簇内密度
    interC = 0  #簇间密度
    #idx_total = []
    K = max(cluster)
    '''
    for i in range(n):  #初始化一个所有索引的列表
        idx_total.append(i)
    '''

    #计算簇内密度
    for i in range(max(cluster)):
        idx = mt.find_list(cluster.tolist(), i+1)   #获取某一簇的点的索引
        if size(idx) > 1:  #计算非单元素簇的umqi指标
            ni = size(idx)
            mi = get_m(S, idx, idx, t)
            #idx_else = list(set(idx_total) - set(idx))
            intraC = intraC + mi/(ni*ni-ni)

    #计算簇间密度
    for i in range(max(cluster)):
        idx = mt.find_list(cluster.tolist(), i+1)   #获取某一簇的点的索引
        if size(idx) > 1:  #计算非单元素簇的umqi指标
            ni = size(idx)
            for j in range(i+1, max(cluster)):
                idx2 = mt.find_list(cluster.tolist(), j+1)
                if size(idx2) > 1:
                    nj = size(idx2)
                    mij = get_m(S, idx, idx2, t)
                    interC = interC + mij/(2*ni*nj)

    umqi = intraC/K - interC*2/(K*(K-1))
    return umqi

def get_m(S, idx1,idx2, t):
    m = 0
    for i in idx1:
        for j in idx2:
            if S[i, j] > 0 and S[i, j] <= t:
                m += 1
    return m

def re_annotation(wlbl, clusters):
    change_wlbl = copy(wlbl)
    label_set = unique(wlbl)
    num_count = [0 for i in range(size(label_set))] #定义一个计数器， 大小为wlbl的标签数如0,1标签，为2
    for i in unique(clusters):
        idx = mt.find_list(clusters.tolist(), i)
        for j in idx:
            for k in range(size(label_set)):    #遍历wlbl中索引为clusters聚类结果的值，进行计数
                if wlbl[j] == label_set[k]:
                    num_count[k] += 1
                    break
        max_index = 0
        max_num = num_count[0]
        for j in range(size(label_set)):   #找到最大的计数与其对应标签在label_set中的索引
            if num_count[j] > max_num:
                max_num = num_count[j]
                max_index = j

        for j in idx:                     #改变change_wlbl中的值
            change_wlbl[j] = label_set[max_index]

    return change_wlbl