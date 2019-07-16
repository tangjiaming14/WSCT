from scipy.cluster.vq import *

def Kmeans(Wt, k):
    features = whiten(Wt)
    centroids,distortion = kmeans(features,k)
    code,distance = vq(features,centroids)
    return code, centroids