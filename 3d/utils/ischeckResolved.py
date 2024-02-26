import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.io import savemat
from utils.cheby import cheby
from utils.vals2coeffs3d import vals2coeffs3d

def ischeckResolved3d(f, func, nd):
  global xxx0, yyy0, zzz0, nstored
  
  nalias = f['n']
  nalias = f['n']
  f['checkvals'] = []
  f['checkcoeffs'] = []
  x0, w0, D0 = cheby(nalias,0)
  x0 = (x0+1)/2
  w0 = w0/2
  xx0, yy0, zz0 = np.meshgrid(x0,x0,x0,indexing='ij') # double check later...
  wx0, wy0, wz0 = np.meshgrid(w0,w0,w0,indexing='ij')
  ww0 = wx0*wy0*wz0
  rint = np.zeros(nd)
  for k in range(len(f['id'])):
    dom = f['domain'][:,k]
    sclx = dom[1] - dom[0]
    scly = dom[3] - dom[2]
    sclz = dom[5] - dom[4]
    xx = sclx * xx0 + dom[0]
    yy = scly * yy0 + dom[2]
    zz = sclz * zz0 + dom[4]
    vals = func(xx, yy, zz)
    vals = vals.reshape(nalias,nalias,nalias,-1)
    f['checkvals'].append(vals)
    coeffs = vals2coeffs3d(vals)
    f['checkcoeffs'].append(coeffs)
    if f['height'][k] == 0:
      rint = rint + ((sclx*scly*sclz)*np.sum(vals**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))
  rint = np.sqrt(rint) # this rint should agree rint from tree build
  
  f['checkerror'] =  np.array([[] for k in range(nd)])
  for k in range(len(f['id'])):
    dom = f['domain'][:,k]
    sclx = dom[1] - dom[0]
    scly = dom[3] - dom[2]
    sclz = dom[5] - dom[4]
    coeffs = f['checkcoeffs'][k]
    erra = np.sqrt(   np.sum((coeffs[-2:,:,:,:])**2, axis=(0, 1, 2)) \
                    + np.sum((coeffs[0:-2,-2:,:,:])**2, axis=(0, 1, 2)) \
                    + np.sum((coeffs[0:-2,0:-2,-2:,:])**2, axis=(0, 1, 2)) )/(nalias**3-(nalias-2)**3)
    erra = erra * np.sqrt((sclx*scly*sclz)) / rint
    f['checkerror'] = np.hstack((f['checkerror'], erra[:,np.newaxis])) 

  ids = f['id'][f['height'] == 0]
  checkerror = f['checkerror'][:,ids]
  return f, checkerror #, err_checkvals

def test_ischeckResolved3d():
  1
  
if __name__ == '__main__':
  test_ischeckResolved3d()