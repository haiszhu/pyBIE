import numpy as np
from scipy.io import savemat

c2vmat_cache = {}

def coeffs2refvals3d(coeffs):
  
  p,_,_,nd = coeffs.shape
  nrefpts = p
  if p in c2vmat_cache:
    c2vmat = c2vmat_cache[p]
  else:
    c2vmat = np.ones((nrefpts, p))
    x = np.linspace(-1,1,p)
    c2vmat[:,1] = x
    for k in range(2, p):
      c2vmat[:,k] = 2*x*c2vmat[:,k-1] - c2vmat[:,k-2]
    
  c2vmat_cache[p] = c2vmat
  
  # value to coeffs map
  tmp1 = np.transpose(np.matmul(c2vmat, coeffs),[1,2,0,3])
  tmp2 = np.transpose(np.matmul(c2vmat, tmp1),[1,2,0,3])
  vals = np.transpose(np.matmul(c2vmat, tmp2),[1,2,0,3])
  return vals

def test_coeffs2refvals3d():
  # coeffs = np.random.rand(16,16,16,1)
  coeffs = np.random.rand(16,16,16,2)
  
  vals = coeffs2refvals3d(coeffs)
  savemat('coeffs2refvals3d.mat', {'coeffs': coeffs, 'vals': vals})
  
if __name__=='__main__':
  test_coeffs2refvals3d()