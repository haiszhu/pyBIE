import numpy as np
from scipy.io import savemat

v2cmat_cache = {}

def vals2coeffs3d(vals):
  global v2cmat_cache
  
  # Get the length of the input:
  p = vals.shape[0]  # vals of dim p x p x p x nd
  # vals = vals.reshape(p,p,p,-1)
  
  if p <= 1:
    # Trivial case (constant)
    coeffs = vals
  else:
    # Use matrix multiplication for all problems ...
    if p in v2cmat_cache:
      v2cmat = v2cmat_cache[p]
    else:
      v2cmat = 2*np.cos(np.pi*(np.arange(1,p+1)-1)[:,None]
                               *(2*(p-np.arange(1,p+1))+1)/(2*p))/p
      v2cmat[0,:] = 1/2*v2cmat[0,:]
        
    v2cmat_cache[p] = v2cmat
      
    # value to coeffs map
    tmp1hat = np.transpose(np.matmul(v2cmat, vals),[1,2,0,3])
    tmp2hat = np.transpose(np.matmul(v2cmat, tmp1hat),[1,2,0,3])
    coeffs  = np.transpose(np.matmul(v2cmat, tmp2hat),[1,2,0,3])
    # coeffs  = np.squeeze(coeffs)
    # coeffs = vals2coeffs3d.F[p]
    
  return coeffs# , tmp1hat, tmp2hat

def test_vals2coeffs3d():
  vals = np.random.rand(16,16,16,1)
  # vals = np.random.rand(16,16,16,2)
  # coeffs, tmp1hat, tmp2hat = vals2coeffs3d(vals)
  # savemat('vals2coeffs3d.mat', {'vals': vals, 'coeffs': coeffs, 'tmp1hat': tmp1hat, 'tmp2hat': tmp2hat})
  coeffs = vals2coeffs3d(vals)
  savemat('vals2coeffs3d.mat', {'vals': vals, 'coeffs': coeffs})
  
if __name__=='__main__':
  test_vals2coeffs3d()