import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

def LapSLPmat(t,s):
# 
# 
# Hai 04/17/21

    d = t['x'] - s['x'].transpose()  # 
    A = -1/(2*np.pi) * np.multiply(np.log(np.abs(d)), np.tile(s['ws'].transpose(),(t['x'].size,1)))
    
    return A

from utils.quadr import quadr
from scipy.io import savemat
def test_LapSLPmat():
    Z = lambda t: (1 + 0.3 * np.cos(5 * t)) * np.exp(1j*t)               # starfish param

    s = {}; s['Z'] = Z; s['p'] = 16; n = 128; s = quadr(s,n,'p','G')
    t = {}; t['x'] = s['x'] + 0.2*s['nx']

    A = LapSLPmat(t,s)
    savemat("testMat.mat", mdict={'A': A})

if __name__ == '__main__':
    test_LapSLPmat()
