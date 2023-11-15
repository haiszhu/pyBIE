#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:41:39 2021

@author: hszhu
"""

import numpy as np
from numpy.fft import fft, ifft

def perispecdiff(f):
    # regardless of input, the output is a column array
    N = f.size
    f = f.reshape((N,1))
    if np.mod(N,2) == 0:
        vec = np.c_[0, [1j*np.arange(1,N/2)], 0, [1j*np.arange(-N/2+1,0)]].reshape(N,1)
    else:
        vec = np.c_[0, [1j*np.arange(1,(N+1)/2)], [1j*np.arange((1-N)/2+1,0)]].reshape(N,1)
    return ifft(fft(f,axis=0) * vec, axis=0)

def test_perispecdiff():
    N = 50
    tj = 2*np.pi/N*np.arange(N).reshape((N,1))
    f = np.sin(3*tj)
    fp = 3*np.cos(3*tj)   # trial periodic function & its deriv
    print(np.linalg.norm(fp - perispecdiff(f)))

if __name__ == '__main__':
    test_perispecdiff()