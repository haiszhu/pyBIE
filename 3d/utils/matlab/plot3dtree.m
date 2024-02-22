function plot3dtree(f,func)
% need to remove func ... later
% only plot 1st func

% 
holdState = ishold();
nplotpts = 800;

h = instantiateSlice3GUI();
handles = guihandles(h);
    
dom = f.domain(:,1);
nSteps = f.nSteps;
numpts = 51;

[xx,yy,zz] = meshgrid(linspace(dom(1), dom(2), numpts), ...
    linspace(dom(3), dom(4), numpts), ...
    linspace(dom(5), dom(6), numpts));
% v = feval(f, xx, yy, zz); % next step
% v = func( xx, yy, zz);
if isempty(func)
  vtmp = f.v;
  v = zeros([numpts numpts numpts]);
  % xx = f.xx; 
  % yy = f.yy;
  % zz = f.zz;
  for k = 1:size(vtmp,2)
    v = v + reshape(vtmp(:,k),[numpts numpts numpts]);
  end
else
  vtmp = func( xx(:), yy(:), zz(:));
  v = reshape(vtmp(:,1),[numpts numpts numpts]);
end
if isreal(v)
    [row,col,tube] = ind2sub(size(v), find(v(:) == max(v(:)), 1, 'last'));
else
    [row, col, tube] = ind2sub(size(v), find(abs(v(:)) == max(abs(v(:))), 1, 'last'));    
end
xslice = xx(row,col,tube); 
yslice = yy(row,col,tube); 
zslice = zz(row,col,tube); 

set(handles.xSlider, 'Min', dom(1));
set(handles.xSlider, 'Max', dom(2));
set(handles.xSlider, 'Value', xslice);

set(handles.ySlider, 'Min', dom(3));
set(handles.ySlider, 'Max', dom(4));
set(handles.ySlider, 'Value', yslice);

set(handles.zSlider, 'Min', dom(5));
set(handles.zSlider, 'Max', dom(6));
set(handles.zSlider, 'Value', zslice);

% nSteps = 15; % number of slices allowed
set(handles.xSlider, 'SliderStep', [1/nSteps , 1 ]);
set(handles.ySlider, 'SliderStep', [1/nSteps , 1 ]);
set(handles.zSlider, 'SliderStep', [1/nSteps , 1 ]);

% Choose default command line output for the slice command:
handles.xx = xx;
handles.yy = yy;
handles.zz = zz;
handles.xslice = xslice;
handles.yslice = yslice;
handles.zslice = zslice;
handles.v = v;

if isreal(v)
    slice(xx, yy, zz, v, xslice, yslice, zslice)
    shading interp
    colorbar
else
    hh = slice(xx, yy, zz, angle(-v), xslice, yslice, zslice); 
    set(hh, 'EdgeColor','none')
    caxis([-pi pi]),
    colormap('hsv')
    axis('equal')     
end

% Plot the function
hold on
ids = leaves(f);

% Plot the boxes
xdata = [f.domain([1 2 2 1 1 1 2 2 1 1], ids) ; nan(1, length(ids)); ... % bottom & top
         f.domain([2 2 2 2 1 1], ids) ; nan(1, length(ids));]; % the rest to complete the box
ydata = [f.domain([3 3 4 4 3 3 3 4 4 3], ids) ; nan(1, length(ids)); ...
         f.domain([3 3 4 4 4 4], ids) ; nan(1, length(ids));];
zdata = [f.domain([5 5 5 5 5 6 6 6 6 6], ids) ; nan(1, length(ids)); ...
         f.domain([5 6 6 5 5 6], ids) ; nan(1, length(ids));];
line('XData', xdata(:), 'YData', ydata(:), 'ZData', zdata(:), 'LineWidth', 1)
hold off

%
handles.xdata = xdata(:);
handles.ydata = ydata(:);
handles.zdata = zdata(:);

% Update handles structure
guidata(h, handles);
handles.output = handles.xSlider;

% Force the figure to clear when another plot is drawn on it so that GUI
% widgets don't linger.  (NB:  This property needs to be reset to 'add' every
% time we change the plot using a slider; otherwise, the slider movement will
% itself clear the figure, which is not what we want.)
set(h, 'NextPlot', 'replacechildren');

% keyboard

end

function ids = leaves(f)
%LEAVES   Get the leaf IDs in a TREEFUN2.
%   LEAVES(F) returns the leaf IDs in the TREEFUN2 F.

ids = f.id(f.height == 0);

end
