import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.io import savemat
from utils.cheby import cheby
from utils.vals2coeffs3d import vals2coeffs3d
from utils.coeffs2checkvals3d import coeffs2checkvals3d
from utils.coeffs2refvals3d import coeffs2refvals3d

def isResolved3d(coeffs, dom, n, tol, vmax, f, checkpts, ifcoeffs, rint):
  global xxx0, yyy0, zzz0, www0, nstored
  
  nalias = n
  nrefpts = n
  _,_,_,nd = coeffs.shape
  
  if 'xxx0' not in globals() or n!=nstored:
    nstored = n
    xxx0, yyy0, zzz0 = np.meshgrid(np.linspace(0,1,nrefpts),np.linspace(0,1,nrefpts),np.linspace(0,1,nrefpts),indexing='ij')
    wxxx0, wyyy0, wzzz0 = np.meshgrid(np.ones(nrefpts)/nstored,np.ones(nrefpts)/nstored,np.ones(nrefpts)/nstored,indexing='ij')
    www0 = wxxx0*wyyy0*wzzz0

  sclx = dom[1] - dom[0]
  scly = dom[3] - dom[2]
  sclz = dom[5] - dom[4]
  
  h = sclx
  eta = 0
  
  if not ifcoeffs:
    xxx = sclx * xxx0 + dom[0]
    yyy = scly * yyy0 + dom[2]
    zzz = sclz * zzz0 + dom[4]
    F = f(xxx.flatten(),yyy.flatten(),zzz.flatten()).reshape(-1,nd).transpose()
    G = coeffs2refvals3d(coeffs)
    erra = np.sqrt(np.sum((G-np.transpose(F.reshape(nd,nrefpts,nrefpts,nrefpts),[1,2,3,0]))**2*www0[:,:,:,np.newaxis], axis=(0, 1, 2)))
    resolved = np.prod(erra < tol * np.sqrt(1/(sclx*scly*sclz))*rint)
  else:
    erra = np.sqrt(   np.sum((coeffs[-1:,:,:,:])**2, axis=(0, 1, 2)) \
                    + np.sum((coeffs[0:-1,-1:,:,:])**2, axis=(0, 1, 2)) \
                    + np.sum((coeffs[0:-1,0:-1,-1:,:])**2, axis=(0, 1, 2)) )/np.sqrt(n**3-(n-1)**3)
    # erra = np.sqrt(   np.sum((coeffs[-2:,:,:,:])**2, axis=(0, 1, 2)) \
    #                 + np.sum((coeffs[0:-2,-2:,:,:])**2, axis=(0, 1, 2)) \
    #                 + np.sum((coeffs[0:-2,0:-2,-2:,:])**2, axis=(0, 1, 2)) )/np.sqrt(n**3-(n-2)**3)
    resolved = np.prod(erra < tol * np.sqrt(1/(sclx*scly*sclz))*rint)
  
  if checkpts.size > 0:
    xxx = 2 * ((checkpts[0,:] - dom[0])/sclx) - 1
    yyy = 2 * ((checkpts[1,:] - dom[2])/scly) - 1
    zzz = 2 * ((checkpts[2,:] - dom[4])/sclz) - 1
    in_region = (xxx >= -1) & (xxx <= 1) \
              & (yyy >= -1) & (yyy <= 1) \
              & (zzz >= -1) & (zzz <= 1)
    if np.sum(in_region) > 0:
      F = f(checkpts[0,in_region],checkpts[1,in_region],checkpts[2,in_region]).transpose()
      G = coeffs2checkvals3d(coeffs, xxx[in_region], yyy[in_region], zzz[in_region])
      err_checkvals = np.max(np.abs(F - G), axis=1)
      for k in range(nd):
        resolved = resolved & (err_checkvals[k] * h**eta < tol * np.maximum(vmax[k],1))
      
  return resolved, erra #, err_checkvals

def test_isResolved3d():
  func = lambda x, y, z: np.exp(-(x**2 + y**2 + z**2) * 5) + \
                         np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 1) + \
                         np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 2)
  # func = lambda x, y, z: [np.exp(-(x**2 + y**2 + z**2) * 5), \
  #                         np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 1), \
  #                         np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 2), \
  #                         np.exp(-((x + 1/4)**2 + (y - 1/5)**2 + (z - 4/5)**2) * 2) ]
  nd = 4
  func = lambda x, y, z: np.array([ np.exp(-(x**2 + y**2 + z**2) * 5), \
                                    np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 10), \
                                    np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 20), \
                                    np.exp(-((x + 1/4)**2 + (y - 1/5)**2 + (z - 4/5)**2) * 2)]).reshape(nd,-1).transpose()
  # nd = 1
  # func = lambda x, y, z: np.array([ np.exp(-(x**2 + y**2 + z**2) * 5)]).reshape(nd,-1).transpose()
  
  dom = np.array([-1, 1, -1, 1, -1, 1])
  f = {
    'domain': np.array([[-1], [1], [-1], [1], [-1], [1]]), 
    'tol': 1.0e-12,
    'nSteps': 15,
    'level': np.array([0]),
    'height': np.array([0]),
    'id': np.array([0]), # 
    'parent': np.array([0]),
    'children': np.zeros((8,1)), 
    'coeffs': [],
    'col': np.array([0]),
    'row': np.array([0]),
    'n': 53,
    # 'checkpts': np.array([]), 
    'checkpts': np.array([[0,    0,    0],
                          [1/2, 1/3,  3/5],
                          [-1/2,-1/3, -3/5]]),
    'rint': np.array([[] for k in range(nd)]),
    'vmax': np.array([[] for k in range(nd)]),
    'ifcoeffs': False
  }
  nalias = f['n']
  x0, w0, D0 = cheby(nalias,0)
  x0 = (x0+1)/2
  w0 = w0/2
  xx0, yy0, zz0 = np.meshgrid(x0,x0,x0,indexing='ij') 
  wx0, wy0, wz0 = np.meshgrid(w0,w0,w0,indexing='ij')
  ww0 = wx0*wy0*wz0
  sclx = dom[1] - dom[0]
  scly = dom[3] - dom[2]
  sclz = dom[5] - dom[4]
  xx = sclx * xx0 + dom[0]
  yy = scly * yy0 + dom[2]
  zz = sclz * zz0 + dom[4]
  vals = func(xx, yy, zz)
  vals = vals.reshape(nalias,nalias,nalias,nd)
  coeffs = vals2coeffs3d(vals)
  rint = np.sqrt((sclx*scly*sclz)*np.sum(vals**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))
  
  f['coeffs'].append(coeffs[0:f['n'],0:f['n'],0:f['n']])
  f['rint'] = np.hstack((f['rint'], rint[:,np.newaxis]))
  f['vmax'] = np.hstack((f['vmax'], np.amax(np.abs(vals), axis=(0, 1, 2))[:,np.newaxis]))

  resolved, erra = isResolved3d(f['coeffs'][0], dom, f['n'], 1e-12, f['vmax'][:,0], func, f['checkpts'], f['ifcoeffs'], rint)
  # resolved = 0
  # erra = 0
  savemat('isResolved.mat', {'resolved': resolved, 'coeffs': f['coeffs'][0], 'rint': f['rint'], 'vmax': f['vmax'], 'erra': erra, 'valsdotww0': vals**2*ww0[:,:,:,np.newaxis]})
  
if __name__ == '__main__':
  test_isResolved3d()