import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.isResolved3d import isResolved3d
from utils.refineBox3d import refineBox3d
from scipy.io import savemat

def buildBreadthFirst3d(f, func):
  id = 0
  while id < len(f['id']):
    dom = f['domain'][:, id]
    resolved, coeffs = isResolved3d(func, dom, f['n'], f['tol'])
    if resolved:
      f['coeffs'].append(coeffs)
      f['height'] = np.append(f['height'], 0)
    else:
      # Split into eight child boxes
      f = refineBox3d(f, id)
      f['height'] = np.append(f['height'], 1)
    id = id + 1
    
  return f

def test_buildBreadthFirst3d():
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
    'checkpts': np.zeros((3, 3)) 
  }
  func = lambda x, y, z: np.exp(-(x**2 + y**2 + z**2) * 50) + \
                         np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 10) + \
                         np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 20)
  f = buildBreadthFirst3d(f, func)
  savemat('buildBreadthFirst3d.mat', {'fdomain': f['domain'], 'flevel': f['level'], 'fchildren': f['children'], 'fid': f['id']})
  
if __name__=='__main__':
  test_buildBreadthFirst3d()