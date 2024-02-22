function h = instantiateSlice3GUI()

% Load up the GUI from the *.fig file.
installDir = treefunroot();
h = openFigInCurrentFigure([installDir '/slice.fig']);
% h = openFigInCurrentFigure(['./slice.fig']);

% Do any required initialization of the handle graphics objects.
G = get(h, 'Children');
for (i = 1:1:length(G))
    if ( isa(G(i), 'matlab.ui.control.UIControl') )
        % Adjust the background colors of the sliders.
        if ( strcmp(G(i).Style, 'slider') )
            if ( isequal(get(G(i), 'BackgroundColor'), ...
                    get(0, 'defaultUicontrolBackgroundColor')) )
                set(G(i), 'BackgroundColor', [.9 .9 .9]);
            end
        end
        % Register callbacks.
        switch ( G(i).Tag )
            case 'xSlider'
                G(i).Callback = @(hObj, data) ...
                    xSlider_Callback(hObj, data, guidata(hObj));
            case 'ySlider'
                G(i).Callback = @(hObj, data) ...
                    ySlider_Callback(hObj, data, guidata(hObj));
            case 'zSlider'
                G(i).Callback = @(hObj, data) ...
                    zSlider_Callback(hObj, data, guidata(hObj));
        end
    end
end

% Store handles to GUI objects so that the callbacks can access them. 
guidata(h, guihandles(h));

end

% --- Executes on xSlider movement.
function xSlider_Callback(hObject, eventdata, handles)
% hObject    handle to xSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

nextPlot = get(hObject.Parent, 'NextPlot');
set(hObject.Parent, 'NextPlot', 'add');

xslice = get(hObject, 'Value');         %returns position of slider
yslice = get(handles.ySlider, 'Value'); %returns position of slider
zslice = get(handles.zSlider, 'Value'); %returns position of slider

% The next slice command clears the title, if there was any. So, get that
% and put it again afterwards.
tit = get(gca(), 'Title');
titText = tit.String;

if ( isreal(handles.v) )
    handles.slice = slice(handles.xx, handles.yy, handles.zz, handles.v, ...
        xslice, yslice, zslice);
    shading interp
    colorbar, 
else
    handles.slice = slice(handles.xx, handles.yy, handles.zz, angle(-handles.v), ...
        xslice, yslice, zslice); 
    set(handles.slice, 'EdgeColor','none')
    caxis([-pi pi]),
    colormap('hsv')
    axis('equal')
end
handles.line = line('XData', handles.xdata(:), 'YData', handles.ydata(:), 'ZData', handles.zdata(:), 'LineWidth', 1);
title(titText)
handles.output = hObject;

set(hObject.Parent, 'NextPlot', nextPlot);

end

function ySlider_Callback(hObject, eventdata, handles)
% --- Executes on ySlider movement.
% hObject    handle to ySlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

nextPlot = get(hObject.Parent, 'NextPlot');
set(hObject.Parent, 'NextPlot', 'add');

yslice = get(hObject, 'Value');         %returns position of slider
xslice = get(handles.xSlider, 'Value'); %returns position of slider
zslice = get(handles.zSlider, 'Value'); %returns position of slider

% The next slice command clears the title, if there was any. So, get that
% and put it again afterwards.
tit = get(gca(), 'Title');
titText = tit.String;

if ( isreal(handles.v) )
    handles.slice = slice(handles.xx, handles.yy, handles.zz, handles.v, ...
        xslice, yslice, zslice);
    shading interp
    colorbar, 
else
    handles.slice = slice(handles.xx, handles.yy, handles.zz, angle(-handles.v), ...
        xslice, yslice, zslice); 
    set(handles.slice, 'EdgeColor','none')
    caxis([-pi pi]),
    colormap('hsv')
    axis('equal')    
end
handles.line = line('XData', handles.xdata(:), 'YData', handles.ydata(:), 'ZData', handles.zdata(:), 'LineWidth', 1);
title(titText)
handles.output = hObject;

set(hObject.Parent, 'NextPlot', nextPlot);

end

function zSlider_Callback(hObject, eventdata, handles)
% --- Executes on zSlider movement.
% hObject    handle to zSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

nextPlot = get(hObject.Parent, 'NextPlot');
set(hObject.Parent, 'NextPlot', 'add');

zslice = get(hObject, 'Value');         %returns position of slider
xslice = get(handles.xSlider, 'Value'); %returns position of slider
yslice = get(handles.ySlider, 'Value'); %returns position of slider

% The next slice command clears the title, if there was any. So, get that
% and put it again afterwards.
tit = get(gca(), 'Title');
titText = tit.String;

if ( isreal(handles.v) )
    handles.slice = slice(handles.xx, handles.yy, handles.zz, handles.v, ...
        xslice, yslice, zslice);
    shading interp
    colorbar
else
    handles.slice = slice(handles.xx, handles.yy, handles.zz, angle(-handles.v), ...
        xslice, yslice, zslice); 
    set(handles.slice, 'EdgeColor','none')
    caxis([-pi pi]),
    colormap('hsv')
    axis('equal')    
end
handles.line = line('XData', handles.xdata, 'YData', handles.ydata, 'ZData', handles.zdata, 'LineWidth', 1);
title(titText)
handles.output = hObject;

set(hObject.Parent, 'NextPlot', nextPlot);

end
