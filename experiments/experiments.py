# ============= Experiments across different algorithms  ================

import numpy as np
import dawa
import l1partition
import os
from numpy.linalg import inv
import networkx as nx

x=np.loadtxt('/Users/akbota/Documents/fall16/Privacy/paper/datasets/Income.txt', dtype=int).tolist()
x=x[1000:2500]

print(x)

Q1 = [ [[1, c, c]] for c in range(len(x)) ]
n=len(x)
m=1
W=np.identity(n)

epsilon=0.001
eps=epsilon/2

def transform_laplace(x, epsilon):
    
    n=len(x)
    scale=np.sum(x)
    
    tmp2 = [0 for i in range(n-2)] + [-1,1]
    P = np.array([tmp2 for i in range(n-2)]).flatten()
    P=np.insert(P,0,1).reshape([n-1, n-1])
    xg = np.dot(inv(P),x[:-1])
    xg_tilde = xg + np.random.laplace(scale=1/epsilon, size=len(xg))

    Wg=P

    counts = np.dot(Wg,xg_tilde)
    counts = np.append(counts, scale-np.sum(counts))

    return counts

def matching(x, epsilon, ratio):
    
    n=len(x)
    G=nx.Graph()
    e = (1-ratio)*epsilon
    K=max(x)*1000

    xlist = ['x'+str(i+1) for i in range(n)]
    listik = [(i, j, -(np.absolute(x[i]-x[j])+1/e   + np.random.laplace(scale=2/(epsilon*ratio/2), size=1)   )+2*K ) for i in range(n) for j in range(i)]
    listik_aux = [(i, i+n, -(1/e + np.random.laplace(scale=2/(epsilon*ratio/2), size=1) ) + K) for i in range(n)]
    G.add_weighted_edges_from(listik)
    G.add_weighted_edges_from(listik_aux)
   
    M1=nx.max_weight_matching(G)

    ind = []
    x1 = []
    
    for i in range(n):
        if i<M1[i]:
            if M1[i]<n:
                ind.append([i, M1[i]])
                x1.append(np.sum([x[i], x[M1[i]]]))
            else:
                ind.append([i])
                x1.append(x[i])
    
    sizes = [len(item) for item in ind]
    
    x_tr = transform_laplace(x1, e)
    
    tmp = [ x_tr[i]/sizes[i] for i in range(len(sizes)) for j in range(sizes[i]) ]
    
    flat_ind = [item for sublist in ind for item in sublist]
   
    x_out = np.array([tmp[j] for i in range(n) for j in range(n) if i==flat_ind[j]])
       
    return x_out
    

def method1(x, epsilon, ratio, consistent=False):
    
    n=len(x)
    scale=np.sum(x)
    Q1 = [ [[1, c, c]] for c in range(len(x)) ]
    
    eps=epsilon/2

    hist1 = l1partition.L1partition(x, eps, ratio, gethist=True)

    n1=len(hist1)

    tmp = [[x[i] for i in range(hist1[j][0], hist1[j][1]+1)] for j in range(n1)]
    x1 = [np.sum(i) for i in tmp]
    sizes1 = [len(i) for i in tmp]


    tmp2 = [0 for i in range(n1-2)] + [-1,1]
    P = np.array([tmp2 for i in range(n1-2)]).flatten()
    P=np.insert(P,0,1).reshape([n1-1, n1-1])
    xg1 = np.dot(inv(P),x1[:-1])
    xg_tilde1 = xg1 + np.random.laplace(scale=1/(epsilon*(1-ratio)), size=len(xg1))

    Wg1=P

    counts = np.dot(Wg1,xg_tilde1)
    counts = np.append(counts, scale-np.sum(counts))
    
    if consistent:
        counts = [0 if i<0 else i for i in counts]

    tmp3 = np.divide(counts, sizes1)
    tmp4 = [ [tmp3[i] for j in range(sizes1[i])] for i in range(len(sizes1)) ]

    noisy_answer = np.array([item for sublist in tmp4 for item in sublist])

    return noisy_answer




err_sum_dawa = 0
err_sum_lap = 0
err_sum_dawa_transform = 0
err_sum_transform_lap = 0
err_sum_match_transf = 0


for i in range(m):
    print("-------------------------"+str(i)+"-------------------------")
    
    #-----------------dawa-----------------
    
    np.random.seed(15)
    x_dawa = dawa.dawa(Q1, x, eps, 0.25)
    err_sum_dawa =  err_sum_dawa + np.sum(np.square(x-x_dawa))/n
    
    
    #---------------laplace----------------

    noise = np.random.laplace(scale=1/eps, size=n)
    x_lap = np.dot(W,x) + noise
    err_sum_lap = err_sum_lap + np.sum(np.square(x-x_lap))/n
    
    
    #--------------method 1----------------

    x_dawa_transform = method1(x, epsilon, 0.5)
    err_sum_dawa_transform = err_sum_dawa_transform + np.sum(np.square(x-x_dawa_transform))/n
    
    
    #---------transformed + laplace--------
    
    x_transform_lap = transform_laplace(x, epsilon)
    err_sum_transform_lap = err_sum_transform_lap + np.sum(np.square(x-x_transform_lap))/n
    
    
    #------matching + transformed----------
    
    x_match_transf = matching(x, epsilon, 0.25)
    err_sum_match_transf = err_sum_match_transf + np.sum(np.square(x-x_match_transf))/n
    
    
    
av_err_dawa = err_sum_dawa/m
av_err_lap = err_sum_lap/m
av_err_dawa_transform = err_sum_dawa_transform/m
av_err_transform_lap = err_sum_transform_lap/m

av_err_match_transf = err_sum_match_transf/m

#print("Average error laplace:")
print(av_err_lap)

#print("Average error dawa:")
print(av_err_dawa)

#print("Average error transformed + laplace:")
print(av_err_transform_lap)

#print("Average error dawa + laplace on transformed workload:")
print(av_err_dawa_transform)

#print("Average error matching + transformed:")
print(av_err_match_transf)
