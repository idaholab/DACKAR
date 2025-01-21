import numpy as np
import stumpy
from stumpy import config
import pandas as pd

from .AnomalyBase import AnomalyBase
from .plotUtils import plot_data

import logging

logger = logging.getLogger(__name__)

DASK_CLIENT_AVAIL = False
try:
  from dask.distributed import Client
  DASK_CLIENT_AVAIL = True
except ImportError:
  logger.error('Importing dask.distributed.Client failed. Parallel calculation will not work!')



class MatrixProfile(AnomalyBase):
  """_summary_

  Args:
      AnomalyBase (_type_): _description_
  """


  def __init__(self, m, normalize='robust', method='normal', approx_percentage=0.1, sub_sequence_normalize=False, excl_zone_denom=4):
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
    self._mp = None # store calculated matrix profile values
    self._avail_method = ['normal', 'parallel', 'approx', 'incremental', 'gpu']
    if method.lower() not in self._avail_method:
      raise IOError(f'Unrecognized calculation method: {method}, please choose from {self._avail_method}')
    self._method = method.lower()
    self._scrump_percentage = approx_percentage
    if self._method == 'gpu':
      raise NotImplementedError('Method "gpu" is not implemented yet!')
    self._current_idx = [] # the index for the last entry of current matrix profile
    self._norm_plot = True


  def _fit(self, X, y=None):
    """perform fitting

    Args:
        X (array-like): (n_samples, n_features)
        y (array-like, optional): ignored, (n_samples, n_features). Defaults to None.
    """
    n_T = X.shape[0]
    if X.shape[1] != 1:
      raise IOError('Multiple dimension time series are provided, this is not supported currently!')

    X_ = X[X.columns[0]]
    if y is not None:
      y_ = y[y.columns[0]]
    else:
      y_ = None

    if n_T <= 10000 and self._method == 'approx':
      self._method = 'normal'
      logger.warning('Reset calculation method form "approx" to "normal" due to small size of time series!')
    if self._method == 'normal':
      self._mp = stumpy.stump(T_A=X_, m=self._m, T_B=y_, normalize=self._sub_norm)
    elif self._method == 'parallel':
      if not DASK_CLIENT_AVAIL:
        raise IOError('Dask is not available, please try to install Dask before use the distributed and parallel implementation')
      with Client() as dask_client:
        self._mp = stumpy.stumped(dask_client, T_A=X_, m=self._m, T_B=y_, normalize=self._sub_norm)
    elif self._method == 'approx':
      self._mp = stumpy.scrump(T_A=X_, m=self._m, T_B=y_, percentage = self._scrump_percentage, pre_scrump=True, normalize=self._sub_norm)
    elif self._method == 'incremental':
      if y is not None:
        logger.warning('The annotated time series will not be used for incremental calculation!')
      self._mp = stumpy.stumpi(T=X_, m=self._m, egress=False, normalize=self._sub_norm)

    self._current_idx.append(n_T-1)


  def _evaluate(self, X):
    """perform evaluation

    Args:
        X (array-like): (n_samples, n_features)
    """
    self._current_idx.append(X.shape[0]+self._current_idx[-1])
    if self._method == 'incremental':
      self._mp.update(X)
    else:
      raise NotImplementedError('Evaluate method is not implemented yet!')


  def plot(self):
    """plot data
    """
    X_transform = self._features.copy()
    if self._norm_plot:
      X_transform = self._scalar.fit_transform(self._features)
      X_transform = pd.DataFrame(X_transform, columns=self._features.columns)
    return plot_data(X_transform, self._mp) # Input time series data should be stored in pandas.DataFrame


  def get_mp(self):
    """get matrix profile value
    """
    if self._mp is not None:
      return self._mp.P_
    return None

  def get_mp_index(self):
    """get matrix profile index
    """
    if self._mp is not None:
      return self._mp.I_
    return None

  def get_mp_left_index(self):
    """get left matrix profile index
    """
    if self._mp is not None:
      return self._mp.left_I_
    return None

  def get_mp_right_index(self):
    """get right matrix profile index
    """
    if self._mp is not None:
      return self._mp.right_I_
    return None
