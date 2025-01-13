import logging
logger = logging.getLogger(__name__)

try:
  from matplotlib import pyplot as plt
  from matplotlib.ticker import FuncFormatter
except ImportError:
  logger.error('Importing matplotlib failed. Plotting will not work.')

try:
  import plotly.graph_objs as go
  from plotly.subplots import make_subplots
except ImportError:
  logger.error('Importing plotly failed. Interactive plots will not work.')


def plot(x, y, ax=None, xlabel='time', ylabel='ds', figsize=(10,6), include_legend=False):
  """_summary_

  Args:
      ax (_type_, optional): _description_. Defaults to None.
      xlabel (str, optional): _description_. Defaults to 'time'.
      ylabel (str, optional): _description_. Defaults to 'ds'.
      figsize (tuple, optional): _description_. Defaults to (10,6).
      include_legend (bool, optional): _description_. Defaults to False.
  """
  user_provided_ax = False if ax is None else True
  if ax is None:
    fig = plt.figure(facecolor='w', figsize=figsize)
    ax = fig.add_subplot(111)
  else:
    fig = ax.get_figure()

  ax.plot(x, y, 'k.')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  if include_legend:
    ax.legend()
  if not user_provided_ax:
    fig.tight_layout()
  return fig

def plot_plotly(x, y, xlabel='time', ylabel='ds', figsize=(900, 600)):
  """_summary_

  Args:
      x (_type_): _description_
      y (_type_): _description_
      figsize (_type_): _description_
      xlabel (str, optional): _description_. Defaults to 'time'.
      ylabel (str, optional): _description_. Defaults to 'ds'.
  """
  line_width = 2
  marker_size = 4

  data = []

  data.append(go.Scatter(name='', x=x, y=y, model='lines'))
  layout = go.Layout(width=figsize[0], height=figsize[1], showlegend=False)
  fig = go.Figure(data=data, layout=layout)

  return fig

