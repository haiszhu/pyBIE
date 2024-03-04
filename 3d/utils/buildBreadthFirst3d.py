import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.isResolved3d import isResolved3d
from utils.refineBox3d import refineBox3d
from utils.refineBox3dv2 import refineBox3dv2
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
  vals = vals.reshape(nalias,nalias,nalias,-1)
  
  _,_,_,nd = vals.shape
  coeffs = vals2coeffs3d(vals)
  rint = np.sqrt((sclx*scly*sclz)*np.sum(vals**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))
  
  f['level']    = np.array([0])
  f['height']   = np.array([0])
  f['id']       = np.array([0], dtype=int)
  f['parent']   = np.array([-1])
  f['children'] = np.zeros((8,1))
  f['coeffs']   = []
  f['col']      = np.array([0])
  f['row']      = np.array([0])                      
  f['rint']     = np.array([[] for k in range(nd)])
  f['vmax']     = np.array([[] for k in range(nd)])
  f['rint0']    = np.array([[] for k in range(nd)])
  
  f['coeffs']   = [coeffs] # is this standard in python? not sure
  f['rint']     = np.hstack((f['rint'], rint[:,np.newaxis]))
  f['vmax']     = np.hstack((f['vmax'], np.amax(np.abs(vals), axis=(0, 1, 2))[:,np.newaxis]))
  
  ifcoeffs      = f['ifcoeffs']
  ifstorecoeffs = f['ifstorecoeffs']
  
  # while loop over level
  resolved, erra = isResolved3d(f['coeffs'][0], f['domain'][:, 0], f['n'], f['tol'], f['vmax'][:,0], func, f['checkpts'], ifcoeffs, rint)
  f['rint0']     = np.hstack((f['rint0'], rint[:,np.newaxis]))
  if resolved:
    return
  else:
    refinecnt = 1
    idl_start = 0
    idl = [0]
  if not ifstorecoeffs or not resolved: # save memory, once isResolved done
    f['coeffs'][0] = []
  
  while refinecnt:
    
    # process previous level unresolved boxes, idl(1), idl(2),....
    domc = np.zeros((6, 8*refinecnt))  # allocate space for all children info of this level's boxes
    coeffsc = [None] * (8*refinecnt)
    rintc = np.zeros((nd, 8*refinecnt))
    rint0c = np.zeros((nd, 8*refinecnt))
    vmaxc = np.zeros((nd, 8*refinecnt))
    parentc = np.zeros(8*refinecnt)
    idc = np.zeros(8*refinecnt)
    levelc = np.zeros(8*refinecnt)
    for k in range(refinecnt): # 1 by 1
      id = idl[k]
      domck, coeffsck, rintck, vmaxck = refineBox3dv2(f['domain'][:, id], f['n'], func)  # just refine the box...
      f['children'][:, id] = idl_start + 8*k + np.arange(1, 9)  # start to know
      f['height'][id] = 1
      # 1 to 8
      cidx = 8*k + np.arange(8)
      domc[:, cidx] = domck
      for i in range(8):
        coeffsc[cidx[i]] = coeffsck[i]
      rintc[:, cidx] = rintck
      vmaxc[:, cidx] = vmaxck
      parentc[cidx] = id * np.ones(8)  # these children's parent id
      idc[cidx] = idl_start + 8*k + np.arange(1, 9)  # children self id
      levelc[cidx] = (f['level'][id]+1) * np.ones(8)
      
    childrenc = np.zeros((8, 8 * refinecnt))  # don't know yet, a bunch of 0s to be concatenated to the end of f.children, will know when next level
    heightc = np.zeros(8 * refinecnt)  # tmp leaf box
    f['domain'] = np.hstack((f['domain'], domc))
    f['coeffs'] += coeffsc # is this standard in python? not sure
    f['rint'] = np.hstack((f['rint'], rintc))
    f['rint0'] = np.hstack((f['rint0'], rint0c))
    f['vmax'] = np.hstack((f['vmax'], vmaxc))
    f['parent'] = np.concatenate((f['parent'], parentc.astype(int)))
    f['id'] = np.concatenate((f['id'], idc.astype(int)))
    f['level'] = np.concatenate(((f['level'], levelc.astype(int))))
    f['children'] = np.hstack((f['children'], childrenc.astype(int)))
    f['height'] = np.concatenate((f['height'], heightc.astype(int)))
    
    # update rint, - num of idl parents contribution + 8*num of idl children contribution
    rint = np.sqrt(rint ** 2 - np.sum(f['rint'][:, idl] ** 2, axis=1) +
                   np.sum(f['rint'][:, idl_start+1:idl_start+8*refinecnt+1] ** 2, axis=1))

    # see if newly created boxes need to be refined next
    unresolvedl = np.ones(8*len(idl), dtype=bool)  # potentially unresolved
    idl0 = np.arange(idl_start+1, idl_start+8*len(idl)+1)  # current level id
    for k in range(8*len(idl)):
      id = idl_start + k + 1
      resolved, erra = isResolved3d(f['coeffs'][id], f['domain'][:, id], f['n'], f['tol'], f['vmax'][:,id], func, f['checkpts'], ifcoeffs, rint)
      f['rint0'][:,id] = rint
      if resolved:
        unresolvedl[k] = False
      if not ifstorecoeffs or not resolved: # save memory, once isResolved done
        f['coeffs'][id] = []
      f['coeffs'][id]
        
    idl_start = idl_start + 8 * refinecnt  # next level starts from here
    idl = idl0[unresolvedl]  # select subset of all potential boxes based on unresolved
    refinecnt = len(idl)

  # # while loop over box (original treefun)
  # id = 0
  # rint = f['rint'][:,0]
  # while id < len(f['id']):
  #   resolved, erra = isResolved3d(f['coeffs'][id], f['domain'][:, id], f['n'], f['tol'], f['vmax'][:,id], func, f['checkpts'], f['ifcoeffs'], rint)
  #   f['rint0'] = np.hstack((f['rint0'], rint[:,np.newaxis]))
  #   if resolved:
  #     f['height'][id] = 0
  #     if not f['ifstorecoeffs']: # save memory
  #       f['coeffs'][id] = []
  #   else:
  #     # Split into eight child boxes
  #     f = refineBox3d(f, id, func)
  #     f['height'][id] = 1
  #     f['coeffs'][id] = []
  #     rint = np.sqrt(rint**2 - f['rint'][:,id]**2 + np.sum(f['rint'][:,-8:]**2, axis=1))
  #   id = id + 1
    
  for k in range(len(f['id']) - 1, -1, -1):
    if not f['height'][k] == 0:
      f['height'][k] = 1 + np.max(f['height'][f['children'][:, k].astype(int)])
  
  return f, rint

def test_buildBreadthFirst3d():
  import time
  
  func = lambda x, y, z: np.exp(-(x**2 + y**2 + z**2) * 50) + \
                         np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 10) + \
                         np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 20)
  
  nd = 4
  func = lambda x, y, z: np.array([ np.exp(-(x**2 + y**2 + z**2) * 500), \
                                    np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 100), \
                                    np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 200), \
                                    np.exp(-((x + 1/4)**2 + (y - 1/5)**2 + (z - 4/5)**2) * 200)]).reshape(nd,-1).transpose()
  ifcoeffs = False
  ifstorecoeffs = True
  
  # parameters
  dom = np.array([-1, 1, -1, 1, -1, 1])
  tol = 1.0e-12
  n = 10
  f = {
    'domain': dom.reshape(-1,1), 
    'tol': tol,
    'nSteps': 15,
    'n': n,
    'checkpts': np.array([[0,    0,    0],
                          [1/2, 1/3,  3/5],
                          [-1/2,-1/3, -3/5]]),
    'ifcoeffs': ifcoeffs,
    'ifstorecoeffs': ifstorecoeffs
  }
  start = time.time()
  f, rint = buildBreadthFirst3d(f, func)
  end = time.time()
  print(end - start)
  savemat('buildBreadthFirst3d.mat', {'rint0': f['rint0'], 'rint': rint, 'fdomain': f['domain'], 'flevel': f['level'], 'fchildren': f['children'], 'fheight': f['height'], 'fid': f['id'], 'frint': f['rint']})
  

  # # install pyscf
  # import numpy
  # import matplotlib.pyplot as plt
  # from mpl_toolkits.mplot3d import Axes3D
  # from pyscf import gto
  # from scipy.io import savemat
  # from utils.buildBreadthFirst3d import buildBreadthFirst3d

  # # basis
  # mol = gto.M(
  #     verbose = 0,
  #     atom = '''
  #     o    0    0.       0.
  #     h    0    -0.757   0.587
  #     h    0    0.757    0.587''',
  #     basis = '6-31g')

  # nd = 13
  # func = lambda x, y, z: numpy.array(mol.eval_gto('GTOval_sph',numpy.column_stack([x.flatten(),y.flatten(),z.flatten()])))

  # # nd = 4
  # # func = lambda x, y, z: numpy.array([ numpy.exp(-(x**2 + y**2 + z**2) * 5), \
  # #                                      numpy.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 10), \
  # #                                      numpy.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 20), \
  # #                                      numpy.exp(-((x + 1/4)**2 + (y - 1/5)**2 + (z - 4/5)**2) * 2)]).reshape(nd,-1).transpose()

  # # initialize tree parameters
  # dom = numpy.array([[-5], [5], [-5], [5], [-5], [5]])
  # f = {
  #     'domain': dom, 
  #     'tol': 1.0e-4,
  #     'nSteps': 15,
  #     'level': numpy.array([0]),
  #     'height': numpy.array([0]),
  #     'id': numpy.array([0]), # 
  #     'parent': numpy.array([0]),
  #     'children': numpy.zeros((8,1)), 
  #     'coeffs': [],
  #     'col': numpy.array([0]),
  #     'row': numpy.array([0]),
  #     'n': 8,
  #     'checkpts': numpy.array([[0,    0,     0],
  #                             [0, -0.757, 0.757],
  #                             [0,  0.587, 0.587]]),
  #     'rint': numpy.array([[] for k in range(nd)]),
  #     'vmax': numpy.array([[] for k in range(nd)])                         
  #   }


  # f, rint = buildBreadthFirst3d(f, func)

  # # compute basis for plotting
  # numpts = 51 # this needs to be consistent with the resolution in plot3dtree.m
  # xx, yy, zz = numpy.meshgrid(numpy.linspace(dom[0,0],dom[1,0],numpts),numpy.linspace(dom[2,0],dom[3,0],numpts),numpy.linspace(dom[4,0],dom[5,0],numpts),indexing='ij')
  # v = func(xx.flatten(),yy.flatten(),zz.flatten())

  # savemat('mytest.mat', {'numpts': numpts, 'v': v, 'xx': xx, 'yy': yy, 'zz': zz, 'rint': rint, 'fdomain': f['domain'], 'flevel': f['level'], 'fchildren': f['children'], 'fheight': f['height'], 'fid': f['id'], 'frint': f['rint'], 'ftol': f['tol'], 'fcheckpts': f['checkpts']})
  
if __name__=='__main__':
  test_buildBreadthFirst3d()