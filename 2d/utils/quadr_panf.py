import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.gauss import gauss
from utils.cheby import cheby
from utils.interpmat import interpmat

def quadr_panf(s, be=2, qntype='G'):
#
#
#
    if not 'p' in s:
        s['p'] = 16
    p = s['p']
    if be == 1:
        sf = s
        if qntype=='G':
            x, w, D = gauss(p)
        else:
            x, w, D = cheby(p)
        sf['xp'] = np.dot(D,sf['x']); sf['xpp'] = np.dot(D,sf['xp']); sf['w'] = w
        sf['sp'] = np.abs(sf['xp']); sf['tang'] = sf['xp']/sf['sp']; sf['nx'] = -1j*sf['tang']
    else:
        sf = {}
        sf['p'] = np.ceil(be*s['p']).astype(int); pf = sf['p']
        Imn = interpmat(p, pf, qntype)
        sf['x'] = np.dot(Imn,s['x'])
        if qntype=='G':
            xx, w, D = gauss(pf)
        else:
            xx, w, D = cheby(pf)
        if not 'Zp' in s:
            if not 'xp' in s:   # numerical differentiation
                sf['xp'] = np.dot(D,sf['x'])
            else:   # interpolate from s
                sf['xp'] = np.dot(Imn,s['xp'])*(s['thi']-s['tlo'])/2
        else:   # direct evaluation
            sf['xp'] = 1/2*(s['thi']-s['tlo'])*s['Zp'](s['tlo'] + (1+xx)/2*(s['thi']-s['tlo']))
        if not 'Zpp' in s:
            if not 'xpp' in s:   # numerical differentiation
                sf['xpp'] = np.dot(D,sf['xp'])
            else:   # interpolate from s
                sf['xpp'] = np.dot(Imn,s['xpp'])*(s['thi']-s['tlo'])/2
        else:    # direct evaluation
            sf['xpp'] = 1/2*(s['thi']-s['tlo'])*s['Zpp'](s['tlo'] + (1+xx)/2*(s['thi']-s['tlo']))
        sf['w'] = w; sf['sp'] = np.abs(sf['xp']); sf['tang'] = sf['xp']/sf['sp']; sf['nx'] = -1j*sf['tang']
        sf['cur'] = - (sf['xpp'].conj()*sf['nx']).real/sf['sp']**2
        sf['ws'] = sf['w']*sf['sp']; sf['wxp'] = sf['w']*sf['xp']
    return sf 


import matplotlib.pyplot as plt
from utils.quadr import quadr
def test_quadr_panf():
    k = 0.3   # curvature of panel
    s = {}
    s['Z']  = lambda t: t + 1j*k*t**2 - 1j*k    # map from [-1,1] to panel, endpts at +-1
    s['Zp'] = lambda t: 1 + 2j*k*t              # deriv, must match s.Z
    s['p'] = 16           # Helsing panel order
    s['tlo'] = np.atleast_1d(-1); s['thi'] = np.atleast_1d(1); s = quadr(s,s['p'],'p','G')    # build one panel
    sf = quadr_panf(s,2,'G')

    plt.plot(sf['x'].real,sf['x'].imag,'.')
    plt.quiver(sf['x'].real, sf['x'].imag, sf['nx'].real, sf['nx'].imag) # plot unit normals
    plt.axis('equal')
    plt.show()

if __name__=='__main__':
    test_quadr_panf()