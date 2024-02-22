import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.io import savemat
from utils.cheby import cheby
from utils.vals2coeffs3d import vals2coeffs3d

def refineBox3d(f, id, func):
  
  global xx0, yy0, zz0, ww0, nstored
  
  nalias = f['n']
  
  if 'xx0' not in globals() or f['n']!=nstored:
    nstored = f['n']
    x0, w0, D0 = cheby(nalias,0)
    x0 = (x0+1)/2
    w0 = w0/2
    xx0, yy0, zz0 = np.meshgrid(x0,x0,x0,indexing='ij') # double check later...
    wx0, wy0, wz0 = np.meshgrid(w0,w0,w0,indexing='ij')
    ww0 = wx0*wy0*wz0
  
  # Split into eight child boxes
  dom = f['domain'][:,id]
  xmid = np.mean(dom[0:2])
  ymid = np.mean(dom[2:4])
  zmid = np.mean(dom[4:6])
  
  cid1 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[dom[0]], [xmid], [dom[2]], [ymid], [dom[4]], [zmid]])))
  f['id'] = np.append(f['id'], cid1)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'],f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  cdom1 = f['domain'][:,cid1]
  csclx1 = cdom1[1] - cdom1[0]
  cscly1 = cdom1[3] - cdom1[2]
  csclz1 = cdom1[5] - cdom1[4]
  cxx1 = csclx1 * xx0 + cdom1[0]
  cyy1 = cscly1 * yy0 + cdom1[2]
  czz1 = csclz1 * zz0 + cdom1[4]
  cvals1 = func(cxx1, cyy1, czz1)
  cvals1 = cvals1.reshape(nalias,nalias,nalias,-1)
  ccoeffs1 = vals2coeffs3d(cvals1)
  f['coeffs'].append(ccoeffs1[0:f['n'],0:f['n'],0:f['n']])
  f['rint'] = np.hstack((f['rint'], np.sqrt((csclx1*cscly1*csclz1)*np.sum(cvals1**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))[:,np.newaxis]))
  f['vmax'] = np.hstack((f['vmax'], np.max(np.abs(cvals1), axis=(0, 1, 2))[:,np.newaxis]))
  # f['coeffs'].append({})
  

  cid2 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[xmid], [dom[1]], [dom[2]], [ymid], [dom[4]], [zmid]])))
  f['id'] = np.append(f['id'], cid2)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  cdom2 = f['domain'][:,cid2]
  csclx2 = cdom2[1] - cdom2[0]
  cscly2 = cdom2[3] - cdom2[2]
  csclz2 = cdom2[5] - cdom2[4]
  cxx2 = csclx2 * xx0 + cdom2[0]
  cyy2 = cscly2 * yy0 + cdom2[2]
  czz2 = csclz2 * zz0 + cdom2[4]
  cvals2 = func(cxx2, cyy2, czz2)
  cvals2 = cvals2.reshape(nalias,nalias,nalias,-1)
  ccoeffs2 = vals2coeffs3d(cvals2)
  f['coeffs'].append(ccoeffs2[0:f['n'],0:f['n'],0:f['n']])
  f['rint'] = np.hstack((f['rint'], np.sqrt((csclx2*cscly2*csclz2)*np.sum(cvals2**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))[:,np.newaxis]))
  f['vmax'] = np.hstack((f['vmax'], np.max(np.abs(cvals2), axis=(0, 1, 2))[:,np.newaxis]))
  # f['coeffs'].append({})

  cid3 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[dom[0]], [xmid], [ymid], [dom[3]], [dom[4]], [zmid]])))
  f['id'] = np.append(f['id'], cid3)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  cdom3 = f['domain'][:,cid3]
  csclx3 = cdom3[1] - cdom3[0]
  cscly3 = cdom3[3] - cdom3[2]
  csclz3 = cdom3[5] - cdom3[4]
  cxx3 = csclx3 * xx0 + cdom3[0]
  cyy3 = cscly3 * yy0 + cdom3[2]
  czz3 = csclz3 * zz0 + cdom3[4]
  cvals3 = func(cxx3, cyy3, czz3)
  cvals3 = cvals3.reshape(nalias,nalias,nalias,-1)
  ccoeffs3 = vals2coeffs3d(cvals3)
  f['coeffs'].append(ccoeffs3[0:f['n'],0:f['n'],0:f['n']])
  f['rint'] = np.hstack((f['rint'], np.sqrt((csclx3*cscly3*csclz3)*np.sum(cvals3**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))[:,np.newaxis]))
  f['vmax'] = np.hstack((f['vmax'], np.max(np.abs(cvals3), axis=(0, 1, 2))[:,np.newaxis]))
  # f['coeffs'].append({})

  cid4 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[xmid], [dom[1]], [ymid], [dom[3]], [dom[4]], [zmid]])))
  f['id'] = np.append(f['id'], cid4)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  cdom4 = f['domain'][:,cid4]
  csclx4 = cdom4[1] - cdom4[0]
  cscly4 = cdom4[3] - cdom4[2]
  csclz4 = cdom4[5] - cdom4[4]
  cxx4 = csclx4 * xx0 + cdom4[0]
  cyy4 = cscly4 * yy0 + cdom4[2]
  czz4 = csclz4 * zz0 + cdom4[4]
  cvals4 = func(cxx4, cyy4, czz4)
  cvals4 = cvals4.reshape(nalias,nalias,nalias,-1)
  ccoeffs4 = vals2coeffs3d(cvals4)
  f['coeffs'].append(ccoeffs4[0:f['n'],0:f['n'],0:f['n']])
  f['rint'] = np.hstack((f['rint'], np.sqrt((csclx4*cscly4*csclz4)*np.sum(cvals4**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))[:,np.newaxis]))
  f['vmax'] = np.hstack((f['vmax'], np.max(np.abs(cvals4), axis=(0, 1, 2))[:,np.newaxis]))
  # f['coeffs'].append({})

  cid5 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[dom[0]], [xmid], [dom[2]], [ymid], [zmid], [dom[5]]])))
  f['id'] = np.append(f['id'], cid5)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  cdom5 = f['domain'][:,cid5]
  csclx5 = cdom5[1] - cdom5[0]
  cscly5 = cdom5[3] - cdom5[2]
  csclz5 = cdom5[5] - cdom5[4]
  cxx5 = csclx5 * xx0 + cdom5[0]
  cyy5 = cscly5 * yy0 + cdom5[2]
  czz5 = csclz5 * zz0 + cdom5[4]
  cvals5 = func(cxx5, cyy5, czz5)
  cvals5 = cvals5.reshape(nalias,nalias,nalias,-1)
  ccoeffs5 = vals2coeffs3d(cvals5)
  f['coeffs'].append(ccoeffs5[0:f['n'],0:f['n'],0:f['n']])
  f['rint'] = np.hstack((f['rint'], np.sqrt((csclx5*cscly5*csclz5)*np.sum(cvals5**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))[:,np.newaxis]))
  f['vmax'] = np.hstack((f['vmax'], np.max(np.abs(cvals5), axis=(0, 1, 2))[:,np.newaxis]))
  # f['coeffs'].append({})

  cid6 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[xmid], [dom[1]], [dom[2]], [ymid], [zmid], [dom[5]]])))
  f['id'] = np.append(f['id'], cid6)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  cdom6 = f['domain'][:,cid6]
  csclx6 = cdom6[1] - cdom6[0]
  cscly6 = cdom6[3] - cdom6[2]
  csclz6 = cdom6[5] - cdom6[4]
  cxx6 = csclx6 * xx0 + cdom6[0]
  cyy6 = cscly6 * yy0 + cdom6[2]
  czz6 = csclz6 * zz0 + cdom6[4]
  cvals6 = func(cxx6, cyy6, czz6)
  cvals6 = cvals6.reshape(nalias,nalias,nalias,-1)
  ccoeffs6 = vals2coeffs3d(cvals6)
  f['coeffs'].append(ccoeffs6[0:f['n'],0:f['n'],0:f['n']])
  f['rint'] = np.hstack((f['rint'], np.sqrt((csclx6*cscly6*csclz6)*np.sum(cvals6**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))[:,np.newaxis]))
  f['vmax'] = np.hstack((f['vmax'], np.max(np.abs(cvals6), axis=(0, 1, 2))[:,np.newaxis]))
  # f['coeffs'].append({})

  cid7 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[dom[0]], [xmid], [ymid], [dom[3]], [zmid], [dom[5]]])))
  f['id'] = np.append(f['id'], cid7)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  cdom7 = f['domain'][:,cid7]
  csclx7 = cdom7[1] - cdom7[0]
  cscly7 = cdom7[3] - cdom7[2]
  csclz7 = cdom7[5] - cdom7[4]
  cxx7 = csclx7 * xx0 + cdom7[0]
  cyy7 = cscly7 * yy0 + cdom7[2]
  czz7 = csclz7 * zz0 + cdom7[4]
  cvals7 = func(cxx7, cyy7, czz7)
  cvals7 = cvals7.reshape(nalias,nalias,nalias,-1)
  ccoeffs7 = vals2coeffs3d(cvals7)
  f['coeffs'].append(ccoeffs7[0:f['n'],0:f['n'],0:f['n']])
  f['rint'] = np.hstack((f['rint'], np.sqrt((csclx7*cscly7*csclz7)*np.sum(cvals7**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))[:,np.newaxis]))
  f['vmax'] = np.hstack((f['vmax'], np.max(np.abs(cvals7), axis=(0, 1, 2))[:,np.newaxis]))
  # f['coeffs'].append({})

  cid8 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[xmid], [dom[1]], [ymid], [dom[3]], [zmid], [dom[5]]])))
  f['id'] = np.append(f['id'], cid8)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  cdom8 = f['domain'][:,cid8]
  csclx8 = cdom8[1] - cdom8[0]
  cscly8 = cdom8[3] - cdom8[2]
  csclz8 = cdom8[5] - cdom8[4]
  cxx8 = csclx8 * xx0 + cdom8[0]
  cyy8 = cscly8 * yy0 + cdom8[2]
  czz8 = csclz8 * zz0 + cdom8[4]
  cvals8 = func(cxx8, cyy8, czz8)
  cvals8 = cvals8.reshape(nalias,nalias,nalias,-1)
  ccoeffs8 = vals2coeffs3d(cvals8)
  f['coeffs'].append(ccoeffs8[0:f['n'],0:f['n'],0:f['n']])
  f['rint'] = np.hstack((f['rint'], np.sqrt((csclx8*cscly8*csclz8)*np.sum(cvals8**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))[:,np.newaxis]))
  f['vmax'] = np.hstack((f['vmax'], np.max(np.abs(cvals8), axis=(0, 1, 2))[:,np.newaxis]))
  # f['coeffs'].append({})

  f['children'][:, id] = [cid1, cid2, cid3, cid4, cid5, cid6, cid7, cid8]
  f['height'][id] = 1
  f['coeffs'][id] = []
  
  return f

def test_refineBox3d():
  func = lambda x, y, z: np.exp(-(x**2 + y**2 + z**2) * 50) + \
                         np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 10) + \
                         np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 20)
  # func = lambda x, y, z: [np.exp(-(x**2 + y**2 + z**2) * 5), \
  #                         np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 1), \
  #                         np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 2), \
  #                         np.exp(-((x + 1/4)**2 + (y - 1/5)**2 + (z - 4/5)**2) * 2) ]
  nd = 4
  func = lambda x, y, z: np.array([ np.exp(-(x**2 + y**2 + z**2) * 5), \
                                    np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 10), \
                                    np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 20), \
                                    np.exp(-((x + 1/4)**2 + (y - 1/5)**2 + (z - 4/5)**2) * 2)]).reshape(nd,-1).transpose()
  f = {
    'domain': np.array([[-1], [1], [-1], [1], [-1], [1]]), 
    'tol': 1.0e-12,
    'nSteps': 15,
    'level': np.array([0]),
    'height': np.array([0]),
    'id': np.array([0]), # 
    'parent': np.array([0]),
    'children': np.zeros((8,1)), 
    'coeffs': [{}],
    'col': np.array([0]),
    'row': np.array([0]),
    'n': 12,
    'checkpts': np.array([[0,    0,    0],
                          [1/2, 1/3,  3/5],
                          [-1/2,-1/3, -3/5]]),
    'rint': np.array([[] for k in range(nd)]),
    'vmax': np.array([[] for k in range(nd)])
  }
  f = refineBox3d(f, 0, func)
  # savemat('refineBox3d.mat', {'fdomain': f['domain'], 'flevel': f['level'], 'fchildren': f['children'], 'fid': f['id'], 'rint': f['rint'], 'vals': f['vals'], 'ww0': f['ww0']})
  savemat('refineBox3d.mat', {'fdomain': f['domain'], 'flevel': f['level'], 'fchildren': f['children'], 'fid': f['id'], 'rint': f['rint'], 'vmax': f['vmax']})
  
if __name__=='__main__':
  test_refineBox3d()
