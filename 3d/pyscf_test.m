% to assist plot of ipynb
% run pyscf_test.ipynb 1st & save tree data
%
% 02/22/24

addpath('utils/matlab/')
load('pyscf_test.mat')

f.domain        = fdomain;
f.tol           = ftol;
f.nSteps        = 1; % for plotting (slider does not work unless import multi v data from python side)
f.level         = flevel;
f.height        = fheight;
f.id            = fid+1;
f.parent        = 0;
f.children      = zeros(8, 1);
f.coeffs        = [];
f.col           = uint64(0);
f.row           = uint64(0);
f.n             = 16; % order
f.checkpts      = fcheckpts;

% for plotting, import func value from data
f.v             = v; 
f.numpts        = numpts;
f.xx            = xx;
f.yy            = yy;
f.zz            = zz;

% dom = f.domain(:,1);
% [xx2,yy2,zz2] = ndgrid(linspace(dom(1), dom(2), numpts), ...
%                 linspace(dom(3), dom(4), numpts), ...
%                 linspace(dom(5), dom(6), numpts)); % to be consistent with numpy.meshgrid indexing='ij'

plot3dtree(f,[])

keyboard