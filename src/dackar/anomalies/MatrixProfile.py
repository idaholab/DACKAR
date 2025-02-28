import numpy as np
import stumpy
from stumpy import config
import pandas as pd

from .AnomalyBase import AnomalyBase
from .plotUtils import plot_data, plot_kdp

import logging

logger = logging.getLogger(__name__)

DASK_CLIENT_AVAIL = False
try:
  from dask.distributed import Client
  DASK_CLIENT_AVAIL = True
except ImportError:
  logger.error('Importing dask.distributed.Client failed. Parallel calculation will not work!')
  logger.info('Try to install "pip install dask distributed"')



class MatrixProfile(AnomalyBase):
  """_summary_

  Args:
      AnomalyBase (_type_): _description_
  """

  def __init__(self, m, normalize='robust', method='normal', kdp=False, approx_percentage=0.1, sub_sequence_normalize=False, excl_zone_denom=4):
    """Constructor
    """
    super().__init__(norm=normalize)
    self._m = m # size of slide window
    self._norm = normalize # perform normalization on times series
    self._sub_norm = sub_sequence_normalize # perform normalization on subsequence time series when calculuate matrix profile distance
    # setup exclusion zone for stumpy: i +/- int(np.ceil(m/excl_zone_denom)), default excl_zone_denom = 4
    # this can avoid nearby subsequences since they are likely highly similar
    # the distance computed in the exclusion zone is set to np.inf before the matrix profile value is extracted
    if excl_zone_denom != 4 and isinstance(excl_zone_denom, int):
      config.STUMPY_EXCL_ZONE_DENOM = excl_zone_denom
    self._mp = {} # store calculated matrix profile values
    self._avail_method = ['normal', 'parallel', 'approx', 'incremental', 'gpu']
    if method.lower() not in self._avail_method:
      raise IOError(f'Unrecognized calculation method: {method}, please choose from {self._avail_method}')
    self._method = method.lower()
    self._scrump_percentage = approx_percentage
    if self._method == 'gpu':
      raise NotImplementedError('Method "gpu" is not implemented yet!')
    self._current_idx = [] # the index for the last entry of current matrix profile
    self._norm_plot = True
    self._compute_kdp = kdp
    self._kdp = {}


  def _fit(self, X, y=None):
    """perform fitting

    Args:
        X (pandas.DataFrame): (n_time_steps, n_features)
        y (pandas.DataFrame, optional): ignored, (n_time_steps, n_features). Defaults to None.
    """
    n_T = X.shape[0]

    for i, var in enumerate(self._xcolumns):
      X_ = X[:,i]
      if y is not None:
        if y.shape[1] == 1:
          y_ = y
        elif y.shape[1] == X.shape[1]:
          y_ = y[:,i]
        else:
          raise IOError('Provided data and annotated data are not aligned! Please check your data.')
      else:
        y_ = None

      self._mp[var] = self._compute_mp(X_, y_)

    self._current_idx.append(n_T-1)

    if self._compute_kdp:
      keys = list(self._mp)
      mps = [self._mp[key].P_ for key in keys]
      mps = np.asarray(mps).T
      kdps = np.sort(mps, axis=1)
      kdps_idx = np.argsort(mps, axis=1)
      kdps = np.flip(kdps, axis=1) # flip, sort from max to min
      kdps_idx = np.flip(kdps_idx, axis=1)
      colm = [f'{k+1}-DP' for k in range(len(keys))]
      idx = self._xindex[:kdps.shape[0]]
      self._kdp['kdp'] = pd.DataFrame(kdps, index=idx, columns=colm) # with dimension (time, variable)
      self._kdp['idx'] = pd.DataFrame(kdps_idx, index=idx, columns=colm)  # with dimension (time, variable)
      self._kdp['var'] = keys



  def _compute_mp(self, X_, y_=None):
    """compute matrix profile

    Args:
        X_ (pandas.DataFrame): (n_time_steps, n_features)
        y_ (pandas.DataFrame, optional): ignored, (n_time_steps, n_features). Defaults to None.
    """
    _mp = None
    if self._method == 'normal':
      _mp = stumpy.stump(T_A=X_, m=self._m, T_B=y_, normalize=self._sub_norm)
    elif self._method == 'parallel':
      if not DASK_CLIENT_AVAIL:
        raise IOError('Dask is not available, please try to install Dask before use the distributed and parallel implementation')
      with Client() as dask_client:
        _mp = stumpy.stumped(dask_client, T_A=X_, m=self._m, T_B=y_, normalize=self._sub_norm)
    elif self._method == 'approx':
      _mp = stumpy.scrump(T_A=X_, m=self._m, T_B=y_, percentage = self._scrump_percentage, pre_scrump=True, normalize=self._sub_norm)
    elif self._method == 'incremental':
      if y_ is not None:
        logger.warning('The annotated time series will not be used for incremental calculation!')
      _mp = stumpy.stumpi(T=X_, m=self._m, egress=False, normalize=self._sub_norm)

    return _mp



  def _evaluate(self, X):
    """perform evaluation

    Args:
        X_ (pandas.DataFrame): (n_time_steps, n_features)
    """
    self._current_idx.append(X.shape[0]+self._current_idx[-1])

    if self._method == 'incremental':
      for j, var in enumerate(self._xcolumns):
        for i in range(X.shape[0]):
          self._mp[var].update(X[i,j])
    else:
      raise NotImplementedError('Evaluate method is not implemented yet!')


  def plot(self):
    """plot data
    """
    X_transform = self._features.copy()
    if self._norm_plot:
      X_transform = self._scalar.fit_transform(self._features)
      X_transform = pd.DataFrame(X_transform, index=self._xindex, columns=self._xcolumns)
    return plot_data(X_transform, self._mp) # Input time series data should be stored in pandas.DataFrame

  def plot_kdp(self):
    """plot data
    """
    return plot_kdp(self._kdp['kdp'])

  def get_mp(self):
    """get matrix profile value
    """
    data = {}
    for var, _mp in self._mp.items():
      data[var] = _mp.P_
    return data

  def get_mp_index(self):
    """get matrix profile index
    """
    data = {}
    for var, _mp in self._mp.items():
      data[var] = _mp.I_
    return data

  def get_mp_left_index(self):
    """get left matrix profile index
    """
    data = {}
    for var, _mp in self._mp.items():
      data[var] = _mp.left_I_
    return data

  def get_mp_right_index(self):
    """get right matrix profile index
    """
    data = {}
    for var, _mp in self._mp.items():
      data[var] = _mp.right_I_
    return data
