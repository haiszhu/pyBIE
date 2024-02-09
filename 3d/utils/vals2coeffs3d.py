import numpy as np
from scipy.io import savemat

def vals2coeffs3d(vals):
  
  # Get the length of the input:
  p = vals.shape[0]  # vals of dim p x p x p
  
  if p <= 1:
    # Trivial case (constant)
    coeffs = vals
  else:
    # Use matrix multiplication for all problems ...
    if 'F' not in vals2coeffs3d.__dict__:
      vals2coeffs3d.F = {}
      
    if p not in vals2coeffs3d.F:
      vals2coeffs3d.F[p] = 2*np.cos(np.pi*(np.arange(1,p+1)-1)[:,None]
                                    *(2*(p-np.arange(1,p+1))+1)/(2*p))/p
      vals2coeffs3d.F[p][0,:] = 1/2*vals2coeffs3d.F[p][0,:]
      
    # value to coeffs map
    tmp1hat = np.tensordot(vals2coeffs3d.F[p],vals,axes=([1],[0]))
    tmp1hat = np.transpose(tmp1hat,[2,0,1])
    tmp2hat = np.tensordot(vals2coeffs3d.F[p],tmp1hat,axes=([1],[0]))
    tmp2hat = np.transpose(tmp2hat,[2,0,1])
    coeffs  = np.tensordot(vals2coeffs3d.F[p],tmp2hat,axes=([1],[0]))
    coeffs  = np.transpose(coeffs,[2,0,1])
    # coeffs = vals2coeffs3d.F[p]
    
  return coeffs

def test_vals2coeffs3d():
  vals = np.random.rand(16,16,16)
  coeffs = vals2coeffs3d(vals)
  savemat('vals2coeffs3d.mat', {'vals': vals, 'coeffs': coeffs})
  
if __name__=='__main__':
  test_vals2coeffs3d()