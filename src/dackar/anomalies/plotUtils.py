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

def plot_data(df, mp, title='Data vs. Matrix Profile', sharex=True, gridspec_kw={'hspace':0}):
  """plot data from pandas dataframe (column-wise)

  Args:
      df (pandas.dataframe): data
      title (str, optional): title for the plot. Defaults to ''.
      sharex (bool, optional): share x-axis. Defaults to True.
      gridspec_kw (dict, optional): grid option for plots. Defaults to {'hspace':0}.
  """
  fig, axs = plt.subplots(df.shape[1]*2, sharex=sharex, gridspec_kw=gridspec_kw, figsize=(8, df.shape[1]*4))
  plt.suptitle(title, fontsize=fontsize)
  ylabel = df.columns

  for i in range(df.shape[1]):
    axs[i*2].set_ylabel(ylabel[i], fontsize=fontsize)
    axs[i*2].set_xlabel('Time', fontsize=fontsize)
    axs[i*2].plot(df.index, df[ylabel[i]], label=ylabel[i])
    j = i*2 + 1
    axs[j].set_ylabel('MP'+' for '+str(ylabel[i]), fontsize=fontsize)
    # axs[i].legend()
    axs[j].set_xlabel('Time', fontsize=fontsize)
    P_ = mp[ylabel[i]].P_
    axs[j].plot(df.index[0:len(P_)], P_, c='orange', label=f'Matrix Profile for {ylabel[i]}')
    # axs[j].legend()

  fig.tight_layout()
  return fig


def plot_kdp(df, title='K-Dimensional Profile', sharex=True, gridspec_kw={'hspace':0}):
  """plot K-dimensional profile from pandas dataframe (column-wise)

  Args:
      df (pandas.dataframe): K-dimensional profile
      title (str, optional): title for the plot. Defaults to ''.
      sharex (bool, optional): share x-axis. Defaults to True.
      gridspec_kw (dict, optional): grid option for plots. Defaults to {'hspace':0}.
  """
  fig, axs = plt.subplots(df.shape[1], sharex=sharex, gridspec_kw=gridspec_kw, figsize=(8, df.shape[1]*2))
  plt.suptitle(title, fontsize=fontsize)
  ylabel = df.columns

  for i in range(df.shape[1]):
    axs[i].set_ylabel(ylabel[i], fontsize=fontsize)
    axs[i].set_xlabel('Time', fontsize=fontsize)
    axs[i].plot(df.index, df[ylabel[i]], label=ylabel[i], c='orange')
  fig.tight_layout()
  return fig


def plot_anomaly(df, mps, anomalies_idx, m, title=None, sharex=True, gridspec_kw={'hspace':0}):
  """Plot anomalies for given data

  Args:
      df (pandas.DataFrame): _description_
      mps (dict): dictionary of matrix profiles of df
      anomalies_idx (dict): dictionary of anomaly indices
      m (int): length of slide window
      title (str, optional): title for the plot. Defaults to None.
      sharex (bool, optional): share x-axis. Defaults to True.
      gridspec_kw (dict, optional): _description_. Defaults to {'hspace':0}.
  """
  num_plots = 2*len(mps)
  fig, axs = plt.subplots(num_plots, sharex=sharex, gridspec_kw=gridspec_kw)

  for i, var_name in enumerate(list(mps.keys())):
    axs[i].set_ylabel(var_name, fontsize=fontsize)
    axs[i].plot(df[var_name])
    axs[i].set_xlabel('Time', fontsize=fontsize)

    j = i+num_plots
    axs[j].set_ylabel('MP'+'_'+var_name, fontsize=fontsize)
    axs[j].plot(mps[var_name], c='orange')
    axs[j].set_xlabel('Time', fontsize=fontsize)
    for idx in anomalies_idx[var_name]:
      axs[i].plot(df[var_name].iloc[idx:idx+m], c='red', linewidth=4)
      axs[i].axvline(x=idx, linestyle='dashed', c='black')
      axs[j].axvline(x=idx, linestyle='dashed', c='black')

  fig.tight_layout()
  return fig

