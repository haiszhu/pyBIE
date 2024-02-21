import numpy as np
from scipy.io import savemat

def coeffs2checkvals3d(coeffs, x, y, z):
  
  p = coeffs.shape[0]
  ncheckpts = np.prod(x.shape)
  
  Evalx = np.ones((ncheckpts, p))
  Evaly = np.ones((ncheckpts, p))
  Evalz = np.ones((ncheckpts, p))

  Evalx[:,1] = x
  Evaly[:,1] = y
  Evalz[:,1] = z
  
  for k in range(2, p):
    Evalx[:,k] = 2*x*Evalx[:,k-1] - Evalx[:,k-2]
    Evaly[:,k] = 2*y*Evaly[:,k-1] - Evaly[:,k-2]
    Evalz[:,k] = 2*z*Evalz[:,k-1] - Evalz[:,k-2]
    
  vals = np.zeros(ncheckpts)
  for k in range(0,ncheckpts):
    tmp1    = np.tensordot(Evalx[k,:], coeffs, axes=([0],[0]))
    tmp2    = np.tensordot(Evaly[k,:],   tmp1, axes=([0],[0]))
    vals[k] = np.tensordot(Evalz[k,:],   tmp2, axes=([0],[0]))
    
  return vals

def test_coeffs2checkvals3d():
  coeffs = np.random.rand(16,16,16)
  checkpts = np.array([[0, 1/2, -1/2],
                       [0, 1/3, -1/3],
                       [0, 3/5, -3/5]])
  x = checkpts[0,:]
  y = checkpts[1,:]
  z = checkpts[2,:]
  vals = coeffs2checkvals3d(coeffs, x, y, z)
  savemat('coeffs2checkvals3d.mat', {'vals': vals, 'coeffs': coeffs, 'x': x, 'y': y, 'z': z})
  
if __name__=='__main__':
  test_coeffs2checkvals3d()