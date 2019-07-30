from pylab import *

ROC_result_path = './result/ROC.pkl'

def rank_order_cluster(S, n):

    RO = copy(S)
    for irow in range(n):              #将距离从小到大排序，计算出RO距离矩阵
        idx = S[irow,:].argsort().tolist()
        for jclomun in range(n):
            temp = idx.index(jclomun)
            RO[irow, jclomun] = temp

    ROC = copy(RO)
    for i in range(n):              #RO矩阵改变成论文中的形式
        for j in range(n):
            k = 0
            temp = 0
            while RO[i, k] != j:
                tt = RO[j, :].tolist().index(RO[i, k])
                temp += tt
                k +=1
            ROC[i, j] = temp

    for i in range(n):             #计算d(a, b),存入RO
        for j in range(n):
            RO[i, j] = (ROC[i,j]+ROC[j,i])/(min(ROC[i,j], ROC[j,i]))

    print(RO)
    try:
        g = open(ROC_result_path, 'wb')            #存储wt生成的距离矩阵S的数据，Wt改变后需要重新计算
        pickle.dump(RO,g)
        g.close()
    except e:
        print('存储错误', e)
    return RO