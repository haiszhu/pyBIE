import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyfmmlib import fmm_part
import numpy as np
# env CC=/usr/local/bin/gcc-13 pip install pyfmmlib  in case you get error on mac

def LapSLPFMM2D(s,if_s,t,if_t,sig,iprec=4):
#
# 
# Hai 04/20/21

    source=np.concatenate((s['x'].real,s['x'].imag),axis=1)
    target=np.concatenate((t['x'].real,t['x'].imag),axis=1)
    charge=s['ws']*sig
    if if_s == 1 and if_t == 0:
        u = fmm_part("p", iprec=iprec, kernel=0, sources=source, \
                    mop_charge=charge, dip_charge=None, target=target, dipvec=None)
        u = -u[:,np.newaxis].real/(2*np.pi); ut = []
    if if_s == 0 and if_t == 1:
        ut = fmm_part("P", iprec=iprec, kernel=0, sources=source, \
                    mop_charge=charge, dip_charge=None, target=target, dipvec=None)
        u = []; ut = -ut[:,np.newaxis].real/(2*np.pi)
    if if_s == 1 and if_t == 1:
        u, ut = fmm_part("pP", iprec=iprec, kernel=0, sources=source, \
                        mop_charge=charge, dip_charge=None, target=target, dipvec=None)
        u = -u[:,np.newaxis].real/(2*np.pi); ut = -ut[:,np.newaxis].real/(2*np.pi)
    return u, ut


from LapSLPmat import LapSLPmat
from LapSLPFMM2D import LapSLPFMM2D
def test_LapSLPFMM2D():
    n = 4000
    sources = np.random.randn(n, 2)
    targ_def = (slice(-3, 3, 20j),)
    targets = np.mgrid[targ_def*2]; targets = targets.reshape(2, -1); targets = targets.T
    mopvec = np.random.randn(n, 1)

    s = {}; s['x'] = sources[:,0] + 1j*sources[:,1]; s['x'] = s['x'][:,np.newaxis]; s['ws'] = np.ones((n,1))
    t = {}; t['x'] = targets[:,0] + 1j*targets[:,1]; t['x'] = t['x'][:,np.newaxis]

    As = LapSLPmat(s,s); As[np.arange(0,n),np.arange(0,n)] = 0
    At = LapSLPmat(t,s) 

    u,ut = LapSLPFMM2D(s,1,t,1,mopvec,4)
    u2 = np.dot(As,mopvec); ut2 = np.dot(At,mopvec)

    print(np.amax(np.abs(ut2-ut))/np.amax(ut))
    print(np.amax(np.abs(u2-u))/np.amax(u))

if __name__ == "__main__":
    test_LapSLPFMM2D()
