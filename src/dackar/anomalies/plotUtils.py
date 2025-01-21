import logging
logger = logging.getLogger(__name__)

import pandas as pd

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

fontsize = '12'


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


def plot_data(df, mp, title='Data vs. Matrix Profile', sharex=True, gridspec_kw={'hspace':0}):
  """plot data from pandas dataframe (column-wise)

  Args:
      df (pandas.dataframe): data
      title (str, optional): title for the plot. Defaults to ''.
      sharex (bool, optional): share x-axis. Defaults to True.
      gridspec_kw (dict, optional): grid option for plots. Defaults to {'hspace':0}.
  """
  fig, axs = plt.subplots(df.shape[1]*2, sharex=sharex, gridspec_kw=gridspec_kw)
  plt.suptitle(title, fontsize=fontsize)
  ylabel = df.columns

  for i in range(df.shape[1]):
    axs[i].set_ylabel(ylabel[i], fontsize=fontsize)
    axs[i].set_xlabel('Time', fontsize=fontsize)
    axs[i].plot(df.index, df[ylabel[i]])
    j = i + df.shape[1]
    axs[j].set_ylabel('MP'+'_'+ylabel[i], fontsize=fontsize)
    axs[j].set_xlabel('Time', fontsize=fontsize)
    axs[j].plot(df.index[0:len(mp)], mp.P_, c='orange')

  fig.tight_layout()
  return fig

def plot_anomaly(df, mps, anomalies_idx, m, plot_mp=False, title=None, sharex=True, gridspec_kw={'hspace':0}):
  """Plot anomalies for given data

  Args:
      df (pandas.DataFrame): _description_
      mps (dict): dictionary of matrix profiles of df
      anomalies_idx (dict): dictionary of anomaly indices
      m (int): length of slide window
      plot_mp (bool, optional): plot matrix profiles
      title (str, optional): title for the plot. Defaults to None.
      sharex (bool, optional): share x-axis. Defaults to True.
      gridspec_kw (dict, optional): _description_. Defaults to {'hspace':0}.
  """
  num_plots = len(mps) if not plot_mp else 2*len(mps)
  fig, axs = plt.subplots(num_plots, sharex=sharex, gridspec_kw=gridspec_kw)

  for i, var_name in enumerate(list(mps.keys())):
    axs[i].set_ylabel(var_name, fontsize=fontsize)
    axs[i].plot(df[var_name])
    axs[i].set_xlabel('Time', fontsize=fontsize)
    if plot_mp:
      j = i+num_plots
      axs[j].set_ylabel('MP'+'_'+var_name, fontsize=fontsize)
      axs[j].plot(mps[var_name], c='orange')
      axs[j].set_xlabel('Time', fontsize=fontsize)
    for idx in anomalies_idx[var_name]:
      axs[i].plot(df[var_name].iloc[idx:idx+m], c='red', linewidth=4)
      axs[i].axvline(x=idx, linestyle='dashed', c='black')
      if plot_mp:
        axs[j].axvline(x=idx, linestyle='dashed', c='black')

  fig.tight_layout()
  return fig

