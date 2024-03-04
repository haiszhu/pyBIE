import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.io import savemat
from utils.cheby import cheby
from utils.isResolved3d import isResolved3d
from utils.refineBox3dv2 import refineBox3dv2
from utils.vals2coeffs3d import vals2coeffs3d
from utils.coeffs2refvals3d import coeffs2refvals3d

def ischeckResolved3d(f, func, nd):
  global xxx0, yyy0, zzz0, www0, nstored
  
  dom = f['domain'][:,0]
  
  # 
  n = f['n']
  nalias = n
  nrefpts = n
  
  if 'xxx0' not in globals() or n!=nstored:
    nstored = n
    xxx0, yyy0, zzz0 = np.meshgrid(np.linspace(0,1,nrefpts),np.linspace(0,1,nrefpts),np.linspace(0,1,nrefpts),indexing='ij')
    wxxx0, wyyy0, wzzz0 = np.meshgrid(np.ones(nrefpts)/nstored,np.ones(nrefpts)/nstored,np.ones(nrefpts)/nstored,indexing='ij')
    www0 = wxxx0*wyyy0*wzzz0
    
  f['checkvals'] = []
  f['checkcoeffs'] = []
  x0, w0, D0 = cheby(nalias,0)
  x0 = (x0+1)/2
  w0 = w0/2
  xx0, yy0, zz0 = np.meshgrid(x0,x0,x0,indexing='ij') # double check later...
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
  
  f['checkerror']   = np.array([[] for k in range(nd)])
  f['checkcoeffs']  = [coeffs]
  f['rint']         = np.array([[] for k in range(nd)])
  f['vmax']         = np.array([[] for k in range(nd)])
  f['rint0']        = np.array([[] for k in range(nd)])
  
  ifstorecoeffs     = f['ifstorecoeffs']
  ifcoeffs          = f['ifcoeffs']
  
  resolved = not np.sum(f['children'][:,0]).astype(bool)
  dom = f['domain'][:,0]
  sclx = dom[1] - dom[0]
  scly = dom[3] - dom[2]
  sclz = dom[5] - dom[4]
  _, erra         = isResolved3d(f['checkcoeffs'][0], dom, f['n'], f['tol'], np.array([]), func, np.array([]), ifcoeffs, rint)
  f['rint0']      = np.hstack((f['rint0'], rint[:,np.newaxis]))
  f['rint']       = np.hstack((f['rint'], rint[:,np.newaxis]))
  erra            = np.sqrt(sclx*scly*sclz)*erra/rint
  f['checkerror'] = np.hstack((f['checkerror'], erra[:,np.newaxis]))
  if resolved: # 1 box...
    return
  else:
    refinecnt = 1
    idl_start = 0
    idl = [0]
  if not ifstorecoeffs or not resolved: # save memory, once isResolved done
    f['checkcoeffs'][0] = []
  
  while refinecnt:
    
    # process previous level unresolved boxes, idl(1), idl(2),....
    checkerrorc = np.zeros((nd,8*refinecnt))
    coeffsc = [None] * (8*refinecnt)
    rintc = np.zeros((nd, 8*refinecnt))
    rint0c = np.zeros((nd, 8*refinecnt))
    vmaxc = np.zeros((nd, 8*refinecnt))
    idc = np.zeros(8*refinecnt)
    for k in range(refinecnt): # 1 by 1
      id = idl[k]
      _, coeffsck, rintck, vmaxck = refineBox3dv2(f['domain'][:, id], f['n'], func)  # just refine the box...
      f['children'][:, id] = idl_start + 8*k + np.arange(1, 9)  # start to know
      f['height'][id]      = 1
      # 1 to 8
      cidx                 = 8*k + np.arange(8)
      for i in range(8):
        coeffsc[cidx[i]]   = coeffsck[i]
      rintc[:, cidx]       = rintck
      vmaxc[:, cidx]       = vmaxck
      idc[cidx]            = idl_start + 8*k + np.arange(1, 9)  # children self id
      
    f['checkerror']        = np.hstack((f['checkerror'], checkerrorc))
    f['checkcoeffs']      += coeffsc # is this standard in python? not sure
    f['rint']              = np.hstack((f['rint'], rintc))
    f['rint0']             = np.hstack((f['rint0'], rint0c))
    f['vmax']              = np.hstack((f['vmax'], vmaxc))
    
    # update rint, - num of idl parents contribution + 8*num of idl children contribution
    rint = np.sqrt(rint ** 2 - np.sum(f['rint'][:, idl] ** 2, axis=1) +
                   np.sum(f['rint'][:, idl_start+1:idl_start+8*refinecnt+1] ** 2, axis=1))

    # see if newly created boxes need to be refined next
    unresolvedl = np.ones(8*len(idl), dtype=bool)  # potentially unresolved
    idl0 = np.arange(idl_start+1, idl_start+8*len(idl)+1)  # current level id
    for k in range(8*len(idl)):
      id = idl_start + k + 1
      resolved = not np.sum(f['children'][:,id]).astype(bool)  # if previously resolved, then no children
      dom = f['domain'][:,id]
      sclx = dom[1] - dom[0]
      scly = dom[3] - dom[2]
      sclz = dom[5] - dom[4]
      _, erra = isResolved3d(f['checkcoeffs'][id], dom, f['n'], f['tol'], np.array([]), func, np.array([]), ifcoeffs, rint)
      f['rint0'][:,id]      = rint
      erra                  = np.sqrt(sclx*scly*sclz)*erra/rint # relative error
      f['checkerror'][:,id] = erra
      if resolved:
        unresolvedl[k] = False
      if not ifstorecoeffs or not resolved: # save memory, once isResolved done
        f['coeffs'][id] = []
        
    idl_start = idl_start + 8 * refinecnt  # next level starts from here
    idl = idl0[unresolvedl]  # select subset of all potential boxes based on unresolved
    refinecnt = len(idl)

  ids = f['id'][f['height'] == 0]
  checkerror = f['checkerror'][:,ids]


  # rint = np.zeros(nd)
  # for k in range(len(f['id'])):
  #   dom = f['domain'][:,k]
  #   sclx = dom[1] - dom[0]
  #   scly = dom[3] - dom[2]
  #   sclz = dom[5] - dom[4]
  #   xx = sclx * xx0 + dom[0]
  #   yy = scly * yy0 + dom[2]
  #   zz = sclz * zz0 + dom[4]
  #   vals = func(xx, yy, zz)
  #   vals = vals.reshape(nalias,nalias,nalias,-1)
  #   f['checkvals'].append(vals)
  #   coeffs = vals2coeffs3d(vals)
  #   f['checkcoeffs'].append(coeffs)
  #   if f['height'][k] == 0:
  #     rint = rint + ((sclx*scly*sclz)*np.sum(vals**2*ww0[:,:,:,np.newaxis], axis=(0, 1, 2)))
  # rint = np.sqrt(rint) # this rint should agree rint from tree build
  
  # f['checkerror'] =  np.array([[] for k in range(nd)])
  # for k in range(len(f['id'])):
  #   dom = f['domain'][:,k]
  #   sclx = dom[1] - dom[0]
  #   scly = dom[3] - dom[2]
  #   sclz = dom[5] - dom[4]
  #   coeffs = f['checkcoeffs'][k]
  #   if not f['ifcoeffs']:
  #     xxx = sclx * xxx0 + dom[0]
  #     yyy = scly * yyy0 + dom[2]
  #     zzz = sclz * zzz0 + dom[4]
  #     F = func(xxx.flatten(),yyy.flatten(),zzz.flatten()).reshape(-1,nd).transpose()
  #     G = coeffs2refvals3d(coeffs)
  #     erra = np.sqrt(np.sum((G-np.transpose(F.reshape(nd,nrefpts,nrefpts,nrefpts),[1,2,3,0]))**2*www0[:,:,:,np.newaxis], axis=(0, 1, 2)))
  #     erra = erra * np.sqrt((sclx*scly*sclz)) / rint
  #   else:
  #     erra = np.sqrt(   np.sum((coeffs[-1:,:,:,:])**2, axis=(0, 1, 2)) \
  #                     + np.sum((coeffs[0:-1,-1:,:,:])**2, axis=(0, 1, 2)) \
  #                     + np.sum((coeffs[0:-1,0:-1,-1:,:])**2, axis=(0, 1, 2)) )/np.sqrt(nalias**3-(nalias-1)**3)
  #     erra = erra * np.sqrt((sclx*scly*sclz)) / rint
  #   f['checkerror'] = np.hstack((f['checkerror'], erra[:,np.newaxis])) 

  # ids = f['id'][f['height'] == 0]
  # checkerror = f['checkerror'][:,ids]
  
  return f, checkerror #, err_checkvals

def test_ischeckResolved3d():
  import numpy
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  from pyscf import gto
  from scipy.io import savemat
  from utils.buildBreadthFirst3d import buildBreadthFirst3d
  from utils.plot3dtree import plot3dtree
  from utils.cheby import cheby
  from utils.vals2coeffs3d import vals2coeffs3d
  from utils.ischeckResolved import ischeckResolved3d
  import time
  
  nd = 4
  func = lambda x, y, z: np.array([ np.exp(-(x**2 + y**2 + z**2) * 500), \
                                    np.exp(-((x - 1/2)**2 + (y - 1/3)**2 + (z - 3/5)**2) * 100), \
                                    np.exp(-((x + 1/2)**2 + (y + 1/3)**2 + (z + 3/5)**2) * 200), \
                                    np.exp(-((x + 1/4)**2 + (y - 1/5)**2 + (z - 4/5)**2) * 200)]).reshape(nd,-1).transpose()
  ifcoeffs = True
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
  
  f, checkerror = ischeckResolved3d(f, func, nd)

  # # basis
  # mol_h2o = gto.M(
  #     verbose = 0,
  #     atom = '''
  #     o    0    0.       0.
  #     h    0    -0.757   0.587
  #     h    0    0.757    0.587''',
  #     # basis = 'ccpvdz')    
  #     basis = 'ccpvtz')

  # dom = numpy.array([-5, 5, -5, 5, -5, 5])
  # tol = 1.0e-4
  # p = 8
  # checkpts = numpy.array([[ 0,    0.,       0.],
  #                         [ 0,   -0.757,    0.587],
  #                         [ 0,    0.757,    0.587]])
  # ifcoeffs = False
  # nd1 = 58 # ccpvtz`
  
  # # if want to resolve all N_orb**2 basis
  # nd = nd1**2
  # def pyscffunc_all(x, y, z, mol):
  #   valstmp = numpy.array(mol.eval_gto('GTOval_sph',numpy.column_stack([x.flatten(),y.flatten(),z.flatten()])))
  #   nrows, ncols = valstmp.shape
  #   vals = numpy.zeros((nrows, ncols**2))
  #   for j in range(ncols):
  #     for k in range(ncols):
  #       vals[:, j*ncols+k] = valstmp[:, j] * valstmp[:, k]
  #   return vals
  # func_all = lambda x, y, z: pyscffunc_all(x, y, z, mol_h2o) # compute func2 on tree3
  # func = func_all
  
  # # if want to resolve 1 proxy func
  # nd = 1
  # def pyscffunc_proxy(x, y, z, mol):
  #   valstmp = numpy.array(mol.eval_gto('GTOval_sph',numpy.column_stack([x.flatten(),y.flatten(),z.flatten()])))
  #   nrows, ncols = valstmp.shape
  #   vals = numpy.zeros((nrows))
  #   for j in range(ncols):
  #     vals = vals + valstmp[:, j]**2
  #   return vals
  # func_proxy = lambda x, y, z: pyscffunc_proxy(x, y, z, mol_h2o)
  # func = func_proxy
  
  # # build tree for func, will move initialization to utils
  # tree3_h2o = {
  #     'domain': dom.reshape(-1,1), 
  #     'tol': tol,
  #     'nSteps': 15,
  #     'n': p,
  #     'checkpts': checkpts.transpose(),
  #     'ifcoeffs': ifcoeffs                           
  #   }
  # numpts = 51 # this needs to be consistent with the resolution in plot3dtree.m
  # tree3_h2o, rint = buildBreadthFirst3d(tree3_h2o, func)
  # xx, yy, zz = numpy.meshgrid(numpy.linspace(dom[0],dom[1],numpts),numpy.linspace(dom[2],dom[3],numpts),numpy.linspace(dom[4],dom[5],numpts),indexing='ij')
  # v3 = func(xx.flatten(),yy.flatten(),zz.flatten())
  # savemat('tree3.mat', {'nd': nd, 'numpts': numpts, 'v': v3, 'xx': xx, 'yy': yy, 'zz': zz, 'rint': rint, 'fdomain': tree3_h2o['domain'], 'fn': tree3_h2o['n'], 'flevel': tree3_h2o['level'], 'fchildren': tree3_h2o['children'], 'fheight': tree3_h2o['height'], 'fid': tree3_h2o['id'], 'frint': tree3_h2o['rint'], 'ftol': tree3_h2o['tol'], 'fcheckpts': tree3_h2o['checkpts']})

  # # check error: nd1**2 num of basis to check...
  # checkfunc = lambda x, y, z: pyscffunc_all(x, y, z, mol_h2o) # compute func2 on tree3
  # tree3_h2o, checkerror = ischeckResolved3d(tree3_h2o, checkfunc, nd1**2)
  # savemat('tree_error.mat', {'tol': tree3_h2o['tol'], 'fcheckerror': tree3_h2o['checkerror'],'fheight': tree3_h2o['height'],'checkerror': checkerror})
  # numpy.max(checkerror)
  
if __name__ == '__main__':
  test_ischeckResolved3d()