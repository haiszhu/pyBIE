% to assist plot of ipynb
%
% 02/22/24

addpath('utils/matlab/')

% tree1
load('tree1.mat')
tree1.domain        = fdomain;
tree1.tol           = ftol;
tree1.nSteps        = 1; % for plotting
tree1.level         = flevel;
tree1.height        = fheight;
tree1.id            = fid+1;
tree1.parent        = 0;
tree1.children      = zeros(8, 1);
tree1.coeffs        = [];
tree1.col           = uint64(0);
tree1.row           = uint64(0);
tree1.n             = fn; % order
tree1.checkpts      = fcheckpts;
tree1.v             = v; % for plotting, import func value from data
tree1.numpts        = numpts;
tree1.xx            = xx;
tree1.yy            = yy;
tree1.zz            = zz;
figure(1),clf,
plot3dtree(tree1,[])
disp(['tree1: ', num2str(sum(tree1.height == 0)), ' boxes ']);

% tree2
load('tree2.mat')
tree2.domain        = fdomain;
tree2.tol           = ftol;
tree2.nSteps        = 1; % for plotting
tree2.level         = flevel;
tree2.height        = fheight;
tree2.id            = fid+1;
tree2.parent        = 0;
tree2.children      = zeros(8, 1);
tree2.coeffs        = [];
tree2.col           = uint64(0);
tree2.row           = uint64(0);
tree2.n             = fn; % order
tree2.checkpts      = fcheckpts;
tree2.v             = v; % for plotting, import func value from data
tree2.numpts        = numpts;
tree2.xx            = xx;
tree2.yy            = yy;
tree2.zz            = zz;
figure(2),clf,
plot3dtree(tree2,[])
disp(['tree2: ', num2str(sum(tree2.height == 0)), ' boxes ']);

% % tree3
load('tree3.mat')
tree3.domain        = fdomain;
tree3.tol           = ftol;
tree3.nSteps        = 1; % for plotting
tree3.level         = flevel;
tree3.height        = fheight;
tree3.id            = fid+1;
tree3.parent        = 0;
tree3.children      = zeros(8, 1);
tree3.coeffs        = [];
tree3.col           = uint64(0);
tree3.row           = uint64(0);
tree3.n             = fn; % order
tree3.checkpts      = fcheckpts;
tree3.v             = v(:); % for plotting, import func value from data
tree3.numpts        = numpts;
tree3.xx            = xx;
tree3.yy            = yy;
tree3.zz            = zz;
figure(3),clf,
plot3dtree(tree3,[])
disp(['tree3: ', num2str(sum(tree3.height == 0)), ' boxes ']);

keyboard