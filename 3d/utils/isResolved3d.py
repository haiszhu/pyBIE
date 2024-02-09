import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.io import savemat
from utils.cheby import cheby
from utils.vals2coeffs3d import vals2coeffs3d

def isResolved3d(f, dom, n=16, tol=1e-12):
  global xx0, yy0, zz0, xxx0, yyy0, zzz0, nstored
  
  nalias = n
  nrefpts = 2*n
  
  if 'xx0' not in globals() or 'xxx0' not in globals() or n!=nstored:
    nstored = n
    x0, w0, D0 = cheby(nalias,0)
    x0 = (x0+1)/2
    xx0, yy0, zz0 = np.meshgrid(x0,x0,x0,indexing='ij') # double check later...
    xxx0, yyy0, zzz0 = np.meshgrid(np.linspace(0,1,nrefpts),np.linspace(0,1,nrefpts),np.linspace(0,1,nrefpts),indexing='ij')

  sclx = dom[1] - dom[0]
  scly = dom[3] - dom[2]
  sclz = dom[5] - dom[4]
  xx = sclx * xx0 + dom[0]
  xxx = sclx * xxx0 + dom[0]
  yy = scly * yy0 + dom[2]
  yyy = scly * yyy0 + dom[2]
  zz = sclz * zz0 + dom[4]
  zzz = sclz * zzz0 + dom[4]
  
  vals = f(xx, yy, zz)
  coeffs = vals2coeffs3d(vals)
  coeffs = coeffs[0:n,0:n,0:n]
  Ex = np.sum(np.abs(coeffs[-2:,:,:]))/(2*n**2)
  Ey = np.sum(np.abs(coeffs[:,-2:,:]))/(2*n**2)
  Ez = np.sum(np.abs(coeffs[:,:,-2:]))/(2*n**2)
  err_cfs = (Ex+Ey+Ez)/3
  
  err = err_cfs
  # err = min(err_cfs, err_vals)
  h = sclx
  eta = 0
  
  vmax = np.max(np.abs(vals))
  
  resolved = (err * h ** eta < tol * max(vmax, 1))
  
  return resolved, coeffs

def test_isResolved3d():
  func = lambda x, y, z: np.exp(-(x**2 + y**2 + z**2) * 50) + \
                         np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 10) + \
                         np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 20)
  dom = np.array([-1, 1, -1, 1, -1, 1])
  resolved, coeffs = isResolved3d(func, dom, 16, 1e-12)
  savemat('isResolved.mat', {'resolved': resolved, 'coeffs': coeffs})
  
if __name__ == '__main__':
  test_isResolved3d()