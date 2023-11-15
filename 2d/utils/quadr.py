#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:36:52 2021

@author: hszhu
"""
import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.perispecdiff import perispecdiff
from utils.gauss import gauss
from utils.cheby import cheby

def quadr(s, N = 0, qtype = 'g', qntype = 'G', xwd = None):
# Set up quadrature & geometry for smooth closed curve  
# two ways of calling:
#               quadr(s, N, 'p', 'G')
#               quadr(s, N)
    if qtype == 'g':
        if N != 0 and ('Z' in s or 'x' in s):
            s['t'] = np.arange(0,N)[:,np.newaxis] * (2*np.pi/N)
            if 'Z' in s:
                s['x'] = s['Z'](s['t'])    # use formula, s['Z'] is a lambda function
            if N != s['x'].size:
                raise Exception("N differs from length of s['x']; that sucks!")
        elif 'x' in s:
            s['x'] = np.array(s['x'])[:, np.newaxis]          # ensure col vec
            N = s['x'].size
            s['t'] = np.arange(0,N)[:, np.newaxis] * (2*np.pi/N) # we don't know the actual params, but choose this
        else:
            raise Exception("Need to provide at least s['Z'] and N, or s['x']. Neither found!")

        if 'Zp' in s:
            s['xp'] = s['Zp'](s['t'])
        else:
            s['xp'] = perispecdiff(s['x'])

        if 'Zpp' in s:
            s['xpp'] = s['Zpp'](s['t'])
        else:
            s['xpp'] = perispecdiff(s['xp'])

        # Now local stuff that derives from x, xp, xpp at each node...
        s['sp'] = np.abs(s['xp'])
        s['tang'] = s['xp']/s['sp']
        s['nx'] = -1j*s['tang']
        s['cur'] = -np.real(np.conj(s['xpp'])*s['nx'])/s['sp']**2  # recall real(conj(a)*b) = "a dot b"
        s['w'] = (2*np.pi/N)*s['sp']
        s['cw'] = (2*np.pi/N)*s['xp']  # complex weights (incl complex speed)
        s['ws'] = s['w']  # weight speed
    elif qtype == 'p':
        if not 'Z' in s:
            raise Exception("Need to provide at least s['Z'] to build panels!")
        if not 'p' in s:
            s['p'] = 16  # default panel order
        p = s['p']
        if 'tpan' in s:
            Np = len(s['tpan'])-1
            s['tlo'] = s['tpan'][0:Np]
            s['thi'] = s['tpan'][1:Np+1]
            N = Np*p
        else:
            Np = np.int_(np.ceil(N/p)); N = p*Np      # Np = num of panels
            if not 'tlo' in s or not 'thi' in s:
                s['tlo'] = np.arange(0,Np)[:,np.newaxis]/Np*2*np.pi
                s['thi'] = np.arange(1,Np+1)[:,np.newaxis]/Np*2*np.pi
        s['Np'] = Np
        s['xlo'] = s['Z'](s['tlo'])  # panel start locs
        s['xhi'] = s['Z'](s['thi'])  # panel end locs
        pt = s['thi'] - s['tlo']  # panel size in parameter
        t = np.zeros(N)[:,np.newaxis]; s['w'] = np.zeros(N)[:,np.newaxis]
        if xwd == None:
            if qntype=='G':
                x, w, D = gauss(p)
            else:
                x, w, D = cheby(p)
        else:
            x = xwd['x']
            w = xwd['w']
            D = xwd['D']
        for i in range(0,Np):
            ii = i*p+np.arange(0,p)  # indices of this panel
            t[ii] = s['tlo'][i] + (1+x)/2*pt[i]; s['w'][ii] = w*pt[i]/2  # nodes weights this panel
        s['t'] = t
        s['x'] = s['Z'](t)  # quadr nodes
        s['xp'] = 0*s['x']; s['xpp'] = 0*s['x'] 
        if 'Zp' in s:
            s['xp'] = s['Zp'](s['t'])
        else:
            for i in range(0,Np):
                ii = i*p+np.arange(0,p)  # indices of this panel
                s['xp'][ii] = D.dot(s['x'][ii]*2/pt[i])
        if 'Zpp' in s:
            s['xpp'] = s['Zpp'](s['t'])
        else:
            for i in range(0,Np):
                ii = i*p+np.arange(0,p)  # indices of this panel
                s['xpp'][ii] = D.dot(s['xp'][ii]*2/pt[i])

        # Now local stuff that derives from x, xp, xpp at each node...
        s['sp'] = np.abs(s['xp'])
        s['tang'] = s['xp']/s['sp']
        s['nx'] = -1j*s['tang']
        s['cur'] = -np.real(np.conj(s['xpp'])*s['nx'])/s['sp']**2  # recall real(conj(a)*b) = "a dot b"
        s['cw'] = s['w']*s['xp']  # complex weights (incl complex speed)
        s['ws'] = s['w']*s['sp']  # weight speed
        s['wxp'] = s['w']*s['xp'] # complex speed weights (Helsing's wzp)
    return s

        


import matplotlib.pyplot as plt

def test_quadr():
    Z = lambda t: (1 + 0.3 * np.cos(5 * t)) * np.exp(1j*t)               # starfish param

    s = {}
    s['Z'] = Z
    n = 100
    s = quadr(s,n,'p')
    plt.plot(np.real(s['x']),np.imag(s['x']))
    plt.quiver(s['x'].real, s['x'].imag, s['nx'].real, s['nx'].imag) # plot unit normals
    plt.axis('equal')
    plt.title("Case 1: both s['Z'] and N input")
    plt.show()

 
    
    # Now check that normals from spectral differentiation are accurate:
    Zp = lambda s: -1.5 * np.sin(5*s) * np.exp(1j*s) + 1j * Z(s)        # Z' formula
    s['Zp'] = Zp
    t = quadr(s,n,'p')
    print('error in the normal vec: ', np.linalg.norm(t['nx']-s['nx']))      # should be small


if __name__=='__main__':
    test_quadr()
        
