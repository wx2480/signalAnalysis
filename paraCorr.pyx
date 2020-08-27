# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: profile=True

from libc.stdlib cimport malloc, free
from cython cimport boundscheck,wraparound,nonecheck,cdivision
from cython.parallel cimport prange
from libc.math cimport isnan,NAN,sqrt
import numpy as np  
cimport numpy as np

ctypedef fused ytype:
    const double
    const float
    const int
    const long

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
@cdivision(True)
cpdef double one_corr(ytype[:] x,ytype[:] y) nogil:
    cdef:
        unsigned int num = x.shape[0], i=0, n = 0, n1 = 0
        double mu_y = 0, var_y =0,mu_x = 0.0,var_x=0.0,cov_xy=0.0,corr, x_delta, x_delta_n, y_delta, y_delta_n, y_delta_nn1

    for i in range(num):
        if isnan(y[i]) or isnan(x[i]):
            continue
        n1 = n
        n += 1
        y_delta = y[i] - mu_y
        y_delta_n = y_delta / n
        y_delta_nn1 = y_delta_n * n1

        var_y += y_delta * y_delta_nn1
        mu_y += y_delta_n        

        x_delta = x[i] - mu_x
        x_delta_n = x_delta / n
        var_x += x_delta * x_delta_n * n1
        mu_x += x_delta_n

        cov_xy += x_delta * y_delta_nn1
    
    corr = sqrt(var_y) * sqrt(var_x)
    if corr==0:
        corr = NAN
    else:
        corr = cov_xy / corr
    return corr

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
@cdivision(True)
def corr(ytype[:,:] x,ytype[:,:] y):
    cdef:
        unsigned int N= y.shape[0],i=0
        double tmp
        double[:] res    
    res = np.empty(N,dtype=np.float64)
    for i in range(N):
        tmp = one_corr(x[i],y[i])
        res[i] = tmp
    return np.asarray(res)

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
@cdivision(True)
cpdef double one_corr_weighted(ytype[:] x,ytype[:] y,ytype[:] w) nogil:
    cdef:
        unsigned int num = x.shape[0], i=0
        double wsum = 0,wsum1 = 0,mu_y = 0, var_y =0,mu_x = 0.0,var_x=0.0,cov_xy=0.0,corr,delta_x,delta_y,delta_x_w,delta_y_w,delta_x_ww1,delta_y_ww1

    for i in range(num):
        if isnan(y[i]) or isnan(x[i]) or isnan(w[i]):
            continue
        wsum1 = wsum
        wsum += w[i]
        delta_y = y[i] - mu_y
        delta_y_w = delta_y * w[i] / wsum
        delta_y_ww1 = delta_y_w * wsum1

        var_y += delta_y * delta_y_ww1
        mu_y += delta_y_w
                        
        delta_x = x[i] - mu_x
        delta_x_w = delta_x * w[i] / wsum
        var_x += delta_x * delta_x_w * wsum1
        mu_x += delta_x_w

        cov_xy += delta_x * delta_y_ww1

    corr = sqrt(var_y) * sqrt(var_x)
    if corr==0:
        corr = NAN
    else:
        corr = cov_xy / corr    
    return corr

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
@cdivision(True)
def corr_weighted(ytype[:,:] x,ytype[:,:] y,ytype[:,:] w):
    cdef:
        unsigned int N= y.shape[0],i=0
        double tmp
        double[:] res    
    res = np.empty(N,dtype=np.float64)
    for i in range(N):
        tmp = one_corr_weighted(x[i],y[i],w[i])
        res[i] = tmp
    return np.asarray(res)