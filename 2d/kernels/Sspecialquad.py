import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

def Sspecialquad(t,s,a,b,side = 'i'):
# SSPECIALQUAD - SLP val+grad close-eval Helsing "special quadrature" matrix
# 
# default side = interior
# Hai 04/16/21, modified based on an equivalent Matlab implementation

    zsc = (b-a)/2; zmid = (b+a)/2 # rescaling factor and midpoint of src segment
    y = np.squeeze((s['x']-zmid)/zsc); x = np.squeeze((t['x']-zmid)/zsc)  # transformed src nodes, targ pts
    N = x.size                            # # of targets
    p = s['x'].size                       # assume panel order is # nodes
    c = (1-(-1)**np.arange(1,p+1))/np.arange(1,p+1)              # Helsing c_k, k = 1..p.
    V = np.ones((p,p))+1j*np.zeros((p,p))
    for k in range(1,p): 
        V[:,k] = V[:,k-1]*y  # Vandermonde mat @ nodes
    P = 1j*np.zeros((p+1,N))      # Build P, Helsing's p_k vectorized on all targs...
    d = 1.1; inr = np.squeeze(np.abs(x)<=d); ifr = np.squeeze(np.abs(x)>d)      # near & far treat separately
    # compute P up to p+1 instead of p as in DLP, since q_k needs them:
    # gam = 1j;  % original choice: branch cut is semicircle behind panel
    gam = np.exp(1j*np.pi/4)  # smaller makes cut closer to panel. barnett 4/17/18
    # note gam = 1 fails, and gam = -1 put cuts in the domain.
    if side == 'e':
        gam = gam.conj()   # note gam is a phase, rots branch cut
    P[0,:] = np.log(gam) + np.log((1-x)/(gam*(-1-x)))  # init p_1 for all targs int

    # upwards recurrence for near targets, faster + more acc than quadr... 
    # (note rotation of cut in log to -Im; so cut in x space is lower unit circle)
    Nn =  np.count_nonzero(inr)
    if Nn != 0:  # Criterion added by Hai Zhu 08/24/16 to ensure inr not empty
        for k in range(0,p):
            P[k+1,inr] = x[inr]*P[k,inr] + c[k] # recursion for p_k
    # fine quadr (no recurrence) for far targets (too inaccurate cf downwards)...
    Nf =  np.count_nonzero(ifr); wxp = np.squeeze(s['wxp']/zsc) # rescaled complex speed weights
    if Nf>0: # Criterion added by Bowei Wu 03/05/15 to ensure ifr not empty
        # backward recursion added by Hai at some point
        P[-1,ifr] = np.sum(np.outer((wxp*(V[:,-1]*y)),np.ones((1,Nf)))/(y[:, np.newaxis]-x[ifr]),axis=0) # int y^p/(y-x)
        for ii in np.arange(p,1,-1):
            P[ii-1,ifr] = (P[ii,ifr]-c[ii-1])/x[ifr] # backward recursion

    Q = 1j*np.zeros((p,N)) # compute q_k from p_k via Helsing 2009 eqn (18)... (p even!)
    # Note a rot ang appears here too...  4/17/18
    # gam = exp(1i*pi/4); % 1i;  % moves a branch arc as in p_1
    # if side == 'e', gam = conj(gam); end   % note gam is a phase, rots branch cut
    Q[::2,:] = P[1::2,:] - np.tile(np.log((1-x)*(-1-x)), (np.ceil(p/2).astype(int),1)) # guessed!
    # (-1)^k, k odd, note each log has branch cut in semicircle from -1 to 1
    Q[1::2,:] = P[2::2,:] - np.tile(np.log(gam) + np.log((1-x)/(gam*(-1-x))),(np.floor(p/2).astype(int),1))  # same cut as for p_1
    # Seems like abs fails - we must be using complex SLP ? :
    #Q(1:2:end,:) = P(2:2:end,:) - repmat(log(abs(1-x.'))+log(abs(-1-x.')),[p/2 1]);
    # (-1)^k, k odd, note each log has branch cut in semicircle from -1 to 1
    #Q(2:2:end,:) = P(3:2:end,:) - repmat(log(abs(1-x.'))-log(abs(-1-x.')),[p/2 1]);
    Q = Q*np.tile(1/np.arange(1,p+1)[:,np.newaxis],(1,N)) # k even
    A = (np.linalg.solve(V.transpose(),Q).transpose() * np.tile((1j*s['nx']).conj().transpose(),(N,1))*zsc).real/(2*np.pi*np.abs(zsc)) # solve for special weights...
    A = A*np.abs(zsc) - np.log(np.abs(zsc))/(2*np.pi)*np.tile((np.abs(s['wxp'])).transpose(),(N,1)) # unscale, yuk

    P = P[:-1,:]  # trim P back to p rows since kernel is like DLP
    Az = np.linalg.solve(V.transpose(),P).transpose() * (1/(2*np.pi)) * np.tile((1j*s['nx']).conj().transpose(),(N,1))
    A1 = Az.real; A2 = -Az.imag

    return A, A1, A2


import matplotlib.pyplot as plt
from scipy.io import savemat
from utils.quadr import quadr
from utils.gauss import gauss
from utils.cheby import cheby
def test_Sspecialquad():
    k = 0.3   # curvature of panel
    s = {}
    s['Z']  = lambda t: t + 1j*k*t**2 - 1j*k    # map from [-1,1] to panel, endpts at +-1
    s['Zp'] = lambda t: 1 + 2j*k*t              # deriv, must match s.Z
    s['p'] = 16           # Helsing panel order
    s['tlo'] = np.atleast_1d(-1); s['thi'] = np.atleast_1d(1); s = quadr(s,s['p'],'p','G')    # build one panel
    sigma = lambda t: np.log(2+t)*np.sin(3*t + 1)     # lap dens (S or D), scalar, real
    t = {}
    dx = 0.01; g = np.arange(-1.3,1.3+dx,dx)   # eval and plot grid
    xx, yy = np.meshgrid(g,g)
    t['x'] = xx.transpose().flatten() + 1j*yy.transpose().flatten()
    # t['x'] = s['x'][[0,1,2]] + 1.2*s['nx'][[0,1,2]]
    
    A, A1, A2 = Sspecialquad(t,s,s['Z'](-1),s['Z'](1),'e')
    savemat("testMat.mat", mdict={'A': A})
    # print(A)
    # plt.plot(s['x'].real,s['x'].imag,'.')
    # plt.plot(t['x'].real,t['x'].imag,'*')
    # plt.quiver(s['x'].real, s['x'].imag, s['nx'].real, s['nx'].imag) # plot unit normals
    # plt.axis('equal')
    # plt.title("Case 1: both s['Z'] and N input")
    # plt.show()
    
    

if __name__ == '__main__':
    test_Sspecialquad()