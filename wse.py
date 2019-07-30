import myTools as mt
import numpy
from scipy import linalg
import liteKmeans

def wse(L, Wt, V, wlbl_values, eta, gamma, lambdas, speedup, optimizer, log_opt):

    #get config
    is_log = mt.get_field(log_opt, 'is_log', True)
    num_clusters = mt.get_field(log_opt, 'num_clusters', 2)
    disp_steps = mt.get_field(log_opt, 'disp_steps', 5)

    #Init
    a0 = 1
    itr = 1000
    n = numpy.size(Wt, 0)
    wlbl_t = numpy.unique(wlbl_values)
    is_converge = False
    t = 0
    obj = numpy.ones((itr,1))*0
    obj1 = numpy.ones((itr,1))*0
    obj2 = numpy.ones((itr,1))*0
    CtA = {}

    while t < itr and is_converge == False:

        [obj[t,0], obj1[t,0], obj2[t,0]] = obj_wse(lambdas, Wt, L, wlbl_values)

        if t > 1:
            if abs(obj[t,0] - obj[t-1,0]) < 1e-5:
                print('Quitting: obj stop changing')
                is_converge = True
                break
            #elif obj[t,0] > obj[t-1,0]:
                #print('Quitting: obj start increasing')
                #is_converge = True
                #break

        # update eta
        while obj[t, 0] > obj_QL(Wt, V, lambdas, L, eta, wlbl_values):
            eta = eta * gamma
            print('updating eta = ', eta)

        Wprev = Wt
        # update V
        V = Wt - eta * (2*L*numpy.matrix(Wt))

        if speedup == 'standard':
            for i in range(numpy.size(wlbl_t)):
                gind = mt.matrix_to_1D(numpy.argwhere(wlbl_values == wlbl_t[0]))
                ng = numpy.size(gind)
                Cg = numpy.identity(ng) - (numpy.ones((ng,ng))/ng)
                temp = 2*lambdas*eta*n / (ng*numpy.size(wlbl_t)*Cg)
                Wt[gind, ] = numpy.identity(ng) + numpy.matrix(V[gind, ])/temp
        elif speedup == 'fast':
            for i in range(numpy.size(wlbl_t)):
                gind = mt.matrix_to_1D(numpy.argwhere(wlbl_values == wlbl_t[0]))
                ng = numpy.size(gind)
                beta = 2*lambdas*eta*n / (ng*numpy.size(wlbl_t))

                # spead up version with closed-form update

                beta_a = beta + 1
                beta_b = -beta / ng
                t1 = numpy.matrix(V[gind, ])
                temp = numpy.matrix(numpy.tile(mt.sum_by_column(t1), (ng, 1)))           #未测试
                Wt[gind, ] = 1/beta_a*numpy.matrix(V[gind, ]) - beta_b/(beta_a*(beta_a+beta_b*ng))*temp

        # 1: reuglard gradient descent update
	    #Wt = Wprev - gamma * Wt;

        # 2: accelerated gradient descent update
        at = 2 / (t+3)
        delta = Wt - Wprev
        Wt = Wt + (1-a0)/a0*at*delta
        a0 = at
        Wt = linalg.orth(Wt)

        # Get cluster results
        if is_log and optimizer == 'agd':
            ct, d = liteKmeans.Kmeans(Wt, num_clusters)
            CtA[t] = ct

        #display
        if t % disp_steps == 0:
            print('Iter#  ',t,' obj=',obj[t,0], ' obj1=',obj1[t,0], ' obj2=',obj2[t,0])

        t = t + 1
        #is_converge = True
    print('WSE: stop at iter = ', t)
    return [CtA, Wt, V, obj, obj1, obj2]


def obj_wse(lambdas, Wt, L, wlbl_values):
    n = numpy.size(Wt, 0)
    m = numpy.size(Wt, 1)
    # compute the objective function
    tem = Wt.T*L
    tem2 = numpy.dot(tem, Wt)
    obj1 = numpy.trace(tem2)
    #print('wse_obj1',obj1)
    wlbl_t = numpy.unique(wlbl_values)
    obj2_ = numpy.ones((numpy.size(wlbl_t,0),1))*0         #n*1,全为0的矩阵
    #print(obj2_)
    #wlbl_values = mt.matrix_to_1D(wlbl_values)      #将二维标签数组n*1变为一位数组n
    for i in range(numpy.size(wlbl_t)):
        gind = numpy.argwhere(wlbl_values == wlbl_t[i])
        ng = numpy.size(gind)
        Cg = numpy.identity(ng) - (numpy.ones((ng,ng))/ng)
        temp = numpy.matrix(Wt[gind, ])                                      #图片数据没有完全放入，超出边界，解决，将test所有数据加入
        temp1 = temp.T * Cg
        trace = numpy.trace(numpy.dot(temp1, temp))
        #print(trace)
        obj2_[i, 0] = n/( numpy.size(wlbl_t)* ng)*trace

    obj2 = sum(obj2_)
    #print('wse_obj2',obj2)
    obj = obj1 + lambdas * obj2
    return obj, obj1, obj2

def obj_QL(Wt, V, lambdas, L, eta, wlbl_values):
    n = numpy.size(V, 0)
    m = numpy.size(V, 1)
    wlbl_t = numpy.unique(wlbl_values)
    obj2_ = numpy.ones((numpy.size(wlbl_t,0),1))*0         #n*1,全为0的矩阵

    #wlbl_values = mt.matrix_to_1D(wlbl_values)

    for i in range(numpy.size(wlbl_t)):
        gind = numpy.argwhere(wlbl_values == wlbl_t[i])
        ng = numpy.size(gind)
        Cg = numpy.identity(ng) - (numpy.ones((ng,ng))/ng)
        #a = V[gind, ].reshape(ng, m)
        #print(a.shape)
        temp = numpy.matrix(V[gind, ].reshape(ng, m))
        temp1 = temp.T * Cg
        trace = numpy.trace(numpy.dot(temp1, temp))
        #print(trace)
        obj2_[i, 0] = n/( numpy.size(wlbl_t)* ng)*trace

    obj2 = sum(obj2_)

    gg1 = numpy.matrix(Wt - V)
    obj = numpy.trace(numpy.matrix(Wt.T)*L*numpy.matrix(Wt)) + \
          numpy.trace( gg1.T* numpy.matrix(2*L*V) ) + mt.sum_diag(gg1.T*gg1) / (2*eta) + lambdas *obj2

    return obj