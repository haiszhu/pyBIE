import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from Sspecialquad import Sspecialquad
from LapSLPmat import LapSLPmat
from utils.quadr_panf import quadr_panf
from utils.interpmat import interpmat
from utils.gauss import gauss

def LapSLPSpecialMat(t,s,side = 'i'):
# special quadrature for Laplace BVP...
# close-far criteria could be modified... 
#
# Hai 04/18/21

    be = 2; qntype='G'
    with np.errstate(divide='ignore'):
        A = LapSLPmat(t,s) 
    ls=0; len=s['p']; x0,_,_=gauss(be*s['p'],0) 
    panlen = np.zeros(s['Np'])
    for k in range(0,s['Np']):
        ss={}
        ss['p']=s['p']; ss['x']=s['x'][ls:ls+len:1]; ss['t']=s['t'][ls:ls+len:1]
        ss['xp']=s['xp'][ls:ls+len:1]; ss['xpp']=s['xpp'][ls:ls+len:1]; ss['sp']=s['sp'][ls:ls+len:1]
        ss['w']=s['w'][ls:ls+len:1]; ss['nx']=s['nx'][ls:ls+len:1]
        ss['ws']=s['ws'][ls:ls+len:1]; ss['wxp']=s['wxp'][ls:ls+len:1]
        ss['tlo']=s['tlo'][k]; ss['thi']=s['thi'][k]

        panlen[k] = np.sum(s['ws'][ls:ls+len:1]) 
        ik = (np.abs(np.squeeze(t['x']) - s['xlo'][k]) + np.abs(np.squeeze(t['x']) - s['xhi'][k])) < 1.7*panlen[k]
        if np.sum(ik)>0:
            tt={}; tt['x']=t['x'][ik]
            ssf = quadr_panf(ss,be,qntype); ssf['t'] = x0 # upsampled panel
            SMat,_,_ = Sspecialquad(tt,ssf,s['xlo'][k],s['xhi'][k],side)
            A[ik,ls:ls+len:1] = np.dot(SMat,interpmat(ss['p'],ssf['p'],qntype))
            
        ls=ls+len
        
    return A


import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from utils.quadr import quadr
from scipy.io import savemat
def test_LapSLPSpecialMat():
    
    Z = lambda t: (1 + 0.3 * np.cos(5 * t)) * np.exp(1j*t)               # starfish param
    s = {}; s['Z'] = Z; s['p'] = 16; s['Np'] = 64; s = quadr(s,s['p']*s['Np'],'p','G')
    
    # target
    nx = 150; gx = (np.arange(1,nx+1)/nx*2-1)*1.5; ny = 150; gy = (np.arange(1,ny+1)/ny*2-1)*1.5 # set up plotting grid
    xx, yy = np.meshgrid(gx,gy); zz = xx+1j*yy

    # exact soln
    alpha = np.linspace(-np.pi/4,np.pi/4,5)
    y_source={}; y_source['x']=0.4*np.exp(1j*alpha)[:,np.newaxis]; y_source['ws']=np.ones((5,1))
    pt_source = np.array([1,-3,1/2,5,-1/3])[:,np.newaxis]
    # fhom = np.NAN*xx
    ta = np.angle(zz)
    idx = (1 + 0.3 * np.cos(5 * ta))<np.absolute(zz)
    t = {}; t['x'] = zz[idx][:,np.newaxis]
    # A = LapSLPmat(t,y_source)
    fhom = np.dot(LapSLPmat(t,y_source),pt_source) # the exact soln 
    # verts = [(0.,0.)]*s['p']*s['np']
    # for k in range(0,s['p']*s['np']):
    #     verts[k] = (np.asscalar(s['x'][k].real),np.asscalar(s['x'][k].imag))
    # path = Path(verts)
    # points = np.concatenate(([xx.flatten()], [yy.flatten()]), axis=0)
    # tmp = ~path.contains_points(points.reshape(nx*ny, 2))
    # notIN = tmp.reshape(nx,ny)

    rhs = np.dot(LapSLPmat(s,y_source),pt_source)
    A = LapSLPSpecialMat(s,s,'e')
    tau = np.linalg.solve(A,rhs)

    u = np.dot(LapSLPSpecialMat(t,s,'e'),tau)
    err = np.amax(np.abs(u-fhom))

    Err = (np.NAN*xx).flatten()
    Err[idx.flatten()] = np.abs(u-fhom).flatten()
    Err = Err.reshape(nx,ny)

    plt.pcolor(xx, yy, np.log10(Err),shading='auto')
    plt.colorbar()
    plt.plot(s['x'].real,s['x'].imag,'.k')

    # A = LapSLPSpecialMat(s,s,'e')
    # savemat("testMat.mat", mdict={'A': A})
    # print(np.abs(u-fhom))
    # print(Err[idx.flatten()])
    # plt.plot(s['x'].real,s['x'].imag,'.')
    # plt.plot(xx[idx],yy[idx],'*')
    plt.show()
    

if __name__ == '__main__':
    test_LapSLPSpecialMat()
