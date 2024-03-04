import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.io import savemat
from utils.cheby import cheby
from utils.vals2coeffs3d import vals2coeffs3d

def refineBox3dv2(dom, n, func):
  
  global xx0, yy0, zz0, ww0, nstored
  
  nalias = n
  
  if 'xx0' not in globals() or n!=nstored:
    nstored = n
    x0, w0, D0 = cheby(nalias,0)
    x0 = (x0+1)/2
    w0 = w0/2
    xx0, yy0, zz0 = np.meshgrid(x0,x0,x0,indexing='ij') # double check later...
    wx0, wy0, wz0 = np.meshgrid(w0,w0,w0,indexing='ij')
    ww0 = wx0*wy0*wz0
  
  # Split into eight child boxes
  xmid = np.mean(dom[0:2])
  ymid = np.mean(dom[2:4])
  zmid = np.mean(dom[4:6])
  
  # domain and coeffs
  domain = np.zeros((6, 8))
  coeffs = [None] * 8
  
  cdom1 = [dom[0], xmid, dom[2], ymid, dom[4], zmid]
  csclx1 = cdom1[1] - cdom1[0]
  cscly1 = cdom1[3] - cdom1[2]
  csclz1 = cdom1[5] - cdom1[4]
  cxx1 = csclx1 * xx0 + cdom1[0]
  cyy1 = cscly1 * yy0 + cdom1[2]
  czz1 = csclz1 * zz0 + cdom1[4]
  cvals1 = func(cxx1, cyy1, czz1)
  cvals1 = cvals1.reshape(nalias,nalias,nalias,-1)
  ccoeffs1 = vals2coeffs3d(cvals1)
  domain[:, 0] = cdom1
  coeffs[0] = ccoeffs1
  
  cdom2 = [xmid, dom[1], dom[2], ymid, dom[4], zmid]
  csclx2 = cdom2[1] - cdom2[0]
  cscly2 = cdom2[3] - cdom2[2]
  csclz2 = cdom2[5] - cdom2[4]
  cxx2 = csclx2 * xx0 + cdom2[0]
  cyy2 = cscly2 * yy0 + cdom2[2]
  czz2 = csclz2 * zz0 + cdom2[4]
  cvals2 = func(cxx2, cyy2, czz2)
  cvals2 = cvals2.reshape(nalias,nalias,nalias,-1)
  ccoeffs2 = vals2coeffs3d(cvals2)
  domain[:, 1] = cdom2
  coeffs[1] = ccoeffs2
  
  cdom3 = [dom[0], xmid, ymid, dom[3], dom[4], zmid]
  csclx3 = cdom3[1] - cdom3[0]
  cscly3 = cdom3[3] - cdom3[2]
  csclz3 = cdom3[5] - cdom3[4]
  cxx3 = csclx3 * xx0 + cdom3[0]
  cyy3 = cscly3 * yy0 + cdom3[2]
  czz3 = csclz3 * zz0 + cdom3[4]
  cvals3 = func(cxx3, cyy3, czz3)
  cvals3 = cvals3.reshape(nalias,nalias,nalias,-1)
  ccoeffs3 = vals2coeffs3d(cvals3)
  domain[:, 2] = cdom3
  coeffs[2] = ccoeffs3
  
  cdom4 = [xmid, dom[1], ymid, dom[3], dom[4], zmid]
  csclx4 = cdom4[1] - cdom4[0]
  cscly4 = cdom4[3] - cdom4[2]
  csclz4 = cdom4[5] - cdom4[4]
  cxx4 = csclx4 * xx0 + cdom4[0]
  cyy4 = cscly4 * yy0 + cdom4[2]
  czz4 = csclz4 * zz0 + cdom4[4]
  cvals4 = func(cxx4, cyy4, czz4)
  cvals4 = cvals4.reshape(nalias,nalias,nalias,-1)
  ccoeffs4 = vals2coeffs3d(cvals4)
  domain[:, 3] = cdom4
  coeffs[3] = ccoeffs4

  cdom5 = [dom[0], xmid, dom[2], ymid, zmid, dom[5]]
  csclx5 = cdom5[1] - cdom5[0]
  cscly5 = cdom5[3] - cdom5[2]
  csclz5 = cdom5[5] - cdom5[4]
  cxx5 = csclx5 * xx0 + cdom5[0]
  cyy5 = cscly5 * yy0 + cdom5[2]
  czz5 = csclz5 * zz0 + cdom5[4]
  cvals5 = func(cxx5, cyy5, czz5)
  cvals5 = cvals5.reshape(nalias,nalias,nalias,-1)
  ccoeffs5 = vals2coeffs3d(cvals5)
  domain[:, 4] = cdom5
  coeffs[4] = ccoeffs5
  
  cdom6 = [xmid, dom[1], dom[2], ymid, zmid, dom[5]]
  csclx6 = cdom6[1] - cdom6[0]
  cscly6 = cdom6[3] - cdom6[2]
  csclz6 = cdom6[5] - cdom6[4]
  cxx6 = csclx6 * xx0 + cdom6[0]
  cyy6 = cscly6 * yy0 + cdom6[2]
  czz6 = csclz6 * zz0 + cdom6[4]
  cvals6 = func(cxx6, cyy6, czz6)
  cvals6 = cvals6.reshape(nalias,nalias,nalias,-1)
  ccoeffs6 = vals2coeffs3d(cvals6)
  domain[:, 5] = cdom6
  coeffs[5] = ccoeffs6[:n, :n, :n, :]

  cdom7 = [dom[0], xmid, ymid, dom[3], zmid, dom[5]]
  csclx7 = cdom7[1] - cdom7[0]
  cscly7 = cdom7[3] - cdom7[2]
  csclz7 = cdom7[5] - cdom7[4]
  cxx7 = csclx7 * xx0 + cdom7[0]
  cyy7 = cscly7 * yy0 + cdom7[2]
  czz7 = csclz7 * zz0 + cdom7[4]
  cvals7 = func(cxx7, cyy7, czz7)
  cvals7 = cvals7.reshape(nalias,nalias,nalias,-1)
  ccoeffs7 = vals2coeffs3d(cvals7)
  domain[:, 6] = cdom7
  coeffs[6] = ccoeffs7[:n, :n, :n, :]

  cdom8 = [xmid, dom[1], ymid, dom[3], zmid, dom[5]]
  csclx8 = cdom8[1] - cdom8[0]
  cscly8 = cdom8[3] - cdom8[2]
  csclz8 = cdom8[5] - cdom8[4]
  cxx8 = csclx8 * xx0 + cdom8[0]
  cyy8 = cscly8 * yy0 + cdom8[2]
  czz8 = csclz8 * zz0 + cdom8[4]
  cvals8 = func(cxx8, cyy8, czz8)
  cvals8 = cvals8.reshape(nalias,nalias,nalias,-1)
  ccoeffs8 = vals2coeffs3d(cvals8)
  domain[:, 7] = cdom8
  coeffs[7] = ccoeffs8[:n, :n, :n, :]

  rint = np.hstack([np.sqrt((csclx * cscly * csclz) * np.sum(cvals ** 2 * ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))[:,np.newaxis] 
                           for csclx, cscly, csclz, cvals in [(csclx1, cscly1, csclz1, cvals1), 
                                                             (csclx2, cscly2, csclz2, cvals2), 
                                                             (csclx3, cscly3, csclz3, cvals3),
                                                             (csclx4, cscly4, csclz4, cvals4),
                                                             (csclx5, cscly5, csclz5, cvals5),
                                                             (csclx6, cscly6, csclz6, cvals6),
                                                             (csclx7, cscly7, csclz7, cvals7),
                                                             (csclx8, cscly8, csclz8, cvals8)]])
  vmax = np.hstack([np.max(np.abs(cvals), axis=(0, 1, 2))[:,np.newaxis]
                           for cvals in [cvals1, cvals2, cvals3, cvals4, cvals5, cvals6, cvals7, cvals8]])
  
  return domain, coeffs, rint, vmax

def test_refineBox3dv2():
  1
  
if __name__=='__main__':
  test_refineBox3dv2()
