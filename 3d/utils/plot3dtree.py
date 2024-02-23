import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.io import savemat
import plotly.graph_objects as go

def plot3dtree(tree):
  
  # hopefully, I get the indexing correct, double check
  ids = tree['id'][tree['height'] == 0]
  xdata = np.vstack([tree['domain'][[[0],[1],[1],[0],[0],[0],[1],[1],[0],[0]],ids],\
                        np.full((1, ids.size), np.nan),\
                        tree['domain'][[[1],[1],[1],[1],[0],[0]], ids],\
                        np.full((1, ids.size), np.nan)])
  xdata = xdata.transpose().flatten()
  ydata = np.vstack([tree['domain'][[[2],[2],[3],[3],[2],[2],[2],[3],[3],[2]],ids],\
                        np.full((1, ids.size), np.nan),\
                        tree['domain'][[[2],[2],[3],[3],[3],[3]], ids],\
                        np.full((1, ids.size), np.nan)])
  ydata = ydata.transpose().flatten()
  zdata = np.vstack([tree['domain'][[[4],[4],[4],[4],[4],[5],[5],[5],[5],[5]],ids],\
                        np.full((1, ids.size), np.nan),\
                        tree['domain'][[[4],[5],[5],[4],[4],[5]], ids],\
                        np.full((1, ids.size), np.nan)])
  zdata = zdata.transpose().flatten()

  fig = go.Figure(data=go.Scatter3d(x=xdata, y=ydata, z=zdata, mode='lines+markers', line=dict(color='blue', width=2), marker=dict(symbol='circle', size=1)))

  fig.update_layout(
    scene=dict(
      xaxis=dict(title='X-axis'),
      yaxis=dict(title='Y-axis'),
      zaxis=dict(title='Z-axis')
    ),
    title='3D Line Plot',
    showlegend=False,
    template='plotly_white'
  )

  fig.show()
  
def test_plot3dtree():
  # not yet implemented
  dom = np.array([-1, 1, -1, 1, -1, 1])
  
if __name__=='__main__':
  test_plot3dtree()