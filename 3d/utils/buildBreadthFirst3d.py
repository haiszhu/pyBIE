import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.isResolved3d import isResolved3d
from utils.refineBox3d import refineBox3d
from utils.cheby import cheby
from utils.vals2coeffs3d import vals2coeffs3d
from utils.coeffs2checkvals3d import coeffs2checkvals3d
from scipy.io import savemat

def buildBreadthFirst3d(f, func):
  
  dom = f['domain'][:,0]
  
  #
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
  coeffs = vals2coeffs3d(vals)
  rint = np.sqrt((sclx*scly*sclz)*np.sum(vals**2*ww0))
  
  f['coeffs'].append(coeffs[0:f['n'],0:f['n'],0:f['n']])
  f['rint'] = np.append(f['rint'], rint)
  f['vmax'] = np.append(f['vmax'], np.max(np.abs(vals)))

  id = 0
  rint = f['rint'][0]
  while id < len(f['id']):
    resolved, erra = isResolved3d(f['coeffs'][id], f['domain'][:, id], f['n'], f['tol'], f['vmax'][id], func, f['checkpts'], rint)
    if resolved:
      f['height'] = np.append(f['height'], 0)
    else:
      # Split into eight child boxes
      f = refineBox3d(f, id, func)
      f['height'] = np.append(f['height'], 1)
      f['coeffs'][id] = []
      rint = np.sqrt(rint**2 - f['rint'][id]**2 + np.sum(f['rint'][-8:]**2))
    id = id + 1
    
  return f

def test_buildBreadthFirst3d():
  func = lambda x, y, z: np.exp(-(x**2 + y**2 + z**2) * 50) + \
                         np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 10) + \
                         np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 20)
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
    'n': 16,
    # 'checkpts': np.array([]), 
    'checkpts': np.array([[0, 1/2, -1/2],
                          [0, 1/3, -1/3],
                          [0, 3/5, -3/5]]),
    'rint': np.array([]),
    'vmax': np.array([])
  }
  
  f = buildBreadthFirst3d(f, func)
  savemat('buildBreadthFirst3d.mat', {'fdomain': f['domain'], 'flevel': f['level'], 'fchildren': f['children'], 'fid': f['id']})
  
if __name__=='__main__':
  test_buildBreadthFirst3d()