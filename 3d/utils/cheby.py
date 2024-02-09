import numpy as np
from numpy.fft import fft, ifft
from numpy.matlib import repmat

def cheby(N,ifD=1):
    theta = np.pi*(2*np.arange(1,N+1)-1)/(2*N)
    x = -np.cos(theta)
    
    l = np.int_(np.floor(N/2)+1)
    K = np.arange(0,N-l+1)
    v = np.concatenate((2*np.exp(1j*np.pi*K/N)/(1-4*K**2), np.zeros(l)))
    w = np.real(ifft(v[0:N] + np.conj(v[N+1:0:-1])))
    
    if ifD == 1:
        X = repmat(x,N,1)
        dX = np.transpose(X)-X
        a = np.prod(dX+np.eye(N),axis=1)
        D = np.outer(a,1/a)/(dX+np.eye(N))
        D = D - np.diag(np.sum(D,axis=1))
    else:
        D = {}
    
    x = x[:,np.newaxis]; w = w[:,np.newaxis]
    
    return x, w, D

def test_cheby():
    x = cheby(12)
    print(x)

if __name__=='__main__':
    test_cheby()