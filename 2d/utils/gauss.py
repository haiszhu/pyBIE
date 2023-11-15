#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:11:57 2021

@author: hszhu
"""

import numpy as np

def gauss(N,ifD=1):
    N=N-1; N1=N+1; N2=N+2
    xu=np.linspace(-1,1,N1)
    y=np.cos((2*np.arange(0,N1)+1)*np.pi/(2*N+2)) + (0.27/N1)*np.sin(np.pi*xu*N/N2)
    L=np.zeros((N1,N2))
    y0=2
    while np.max(np.abs(y-y0))>np.finfo(float).eps:
        L[:,0]=1; L[:,1]=y 
        for k in range(2,N1+1): 
            L[:,k]=( (2*k-1)*y*L[:,k-1]-(k-1)*L[:,k-2] )/k
            
        Lp=N2*( L[:,N1-1]-y*L[:,N2-1] )/(1-y**2)
        y0=y
        y=y0-L[:,N2-1]/Lp
    x=y[::-1]
    w=2/((1-y**2)*Lp**2)*(N2/N1)**2
    
    if ifD == 1:
        N = N1
        index = np.arange(0,N)
        D = np.zeros((N,N)); a = np.zeros(N)
        for k in range(0,N):
            notk = index!=k
            a[k] = np.prod(x[k]-x[notk])
            
        for k in range(0,N):
            notk = index!=k
            D[notk,k] = (a[notk]/a[k])/(x[notk]-x[k])
            D[k,k] = np.sum(1/(x[k]-x[notk]))
    else:
        D = {}
    
    x = x[:,np.newaxis]; w = w[:,np.newaxis]

    return x, w, D

def test_gauss():
    x,w,D = gauss(10)
    print(x)
    print(w)

if __name__=='__main__':
    test_gauss()