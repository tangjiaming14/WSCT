import myTools as mt
import numpy

def wse(L, Wt, V, wlbl_values, eta, gamma, lambdas, speedup, optimizer, log_opt):

    #get config
    is_log = mt.get_field(log_opt, 'is_log', False)
    num_clusters = mt.get_field(log_opt, 'num_clusters', 2)
    disp_steps = mt.get_field(log_opt, 'disp_steps', 1)

    #Init
    a0 = 1
    itr = 1000
    n = numpy.size(Wt, 0)
    wlbl_t = numpy.unique(wlbl_values)
    is_converge = False
    t = 1
    obj = numpy.ones((itr,1))*0
    obj1 = numpy.ones((itr,1))*0
    obj2 = numpy.ones((itr,1))*0
    CtA = {}

    while t < itr and is_converge == False:
        [obj[t,0], obj1[t,0], obj2[t,0]] = obj_wse(lambdas, Wt, L, wlbl_values)
        is_converge = True

    return [CtA, Wt, V, obj, obj1, obj2]


def obj_wse(lambdas, Wt, L, wlbl_values):
    n = numpy.size(Wt, 0)

    # compute the objective function
    tem = Wt.T*L
    tem2 = numpy.dot(tem, Wt)
    obj1 = numpy.trace(tem2)
    #print(obj1)
    wlbl_t = numpy.unique(wlbl_values)
    obj2_ = numpy.ones((numpy.size(wlbl_t,0),1))*0         #n*1,全为0的矩阵
    #print(obj2_)
    wlbl_values = mt.matrix_to_1D(wlbl_values)      #将二维标签数组n*1变为一位数组n
    for i in range(numpy.size(wlbl_t)):
        gind = numpy.argwhere(wlbl_values == wlbl_t[i])
        ng = numpy.size(gind)
        Cg = numpy.identity(ng) - (numpy.ones((ng,ng))/ng)
        temp = Wt[gind, ]                                       #图片数据没有完全放入，超出边界，解决，将test所有数据加入
        temp1 = temp.T * Cg
        obj2_[i, 0] = n/( numpy.size(wlbl_t)* ng)*numpy.trace(numpy.dot(temp1, temp))

    return 1, 1, 1