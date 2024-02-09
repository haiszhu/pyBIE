import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.io import savemat

def refineBox3d(f, id):

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
  f['coeffs'].append({})

  cid2 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[xmid], [dom[1]], [dom[2]], [ymid], [dom[4]], [zmid]])))
  f['id'] = np.append(f['id'], cid2)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  f['coeffs'].append({})

  cid3 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[dom[0]], [xmid], [ymid], [dom[3]], [dom[4]], [zmid]])))
  f['id'] = np.append(f['id'], cid3)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  f['coeffs'].append({})

  cid4 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[xmid], [dom[1]], [ymid], [dom[3]], [dom[4]], [zmid]])))
  f['id'] = np.append(f['id'], cid4)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  f['coeffs'].append({})

  cid5 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[dom[0]], [xmid], [dom[2]], [ymid], [zmid], [dom[5]]])))
  f['id'] = np.append(f['id'], cid5)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  f['coeffs'].append({})

  cid6 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[xmid], [dom[1]], [dom[2]], [ymid], [zmid], [dom[5]]])))
  f['id'] = np.append(f['id'], cid6)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  f['coeffs'].append({})

  cid7 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[dom[0]], [xmid], [ymid], [dom[3]], [zmid], [dom[5]]])))
  f['id'] = np.append(f['id'], cid7)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  f['coeffs'].append({})

  cid8 = len(f['id'])
  f['domain'] = np.hstack((f['domain'], np.array([[xmid], [dom[1]], [ymid], [dom[3]], [zmid], [dom[5]]])))
  f['id'] = np.append(f['id'], cid8)
  f['parent'] = np.append(f['parent'], id)
  f['children'] = np.hstack((f['children'], np.zeros((8,1))))
  f['level'] = np.append(f['level'], f['level'][id] + 1)
  f['height'] = np.append(f['height'], 0)
  f['coeffs'].append({})

  f['children'][:, id] = [cid1, cid2, cid3, cid4, cid5, cid6, cid7, cid8]
  f['height'][id] = 1
  f['coeffs'][id] = []
  
  return f

def test_refineBox3d():
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
  f = refineBox3d(f, 0)
  savemat('refineBox3d.mat', {'fdomain': f['domain'], 'flevel': f['level'], 'fchildren': f['children'], 'fid': f['id']})
  
if __name__=='__main__':
  test_refineBox3d()
