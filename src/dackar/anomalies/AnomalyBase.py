"""
  Created on Dec. 19, 2024

  @author: wangc, mandd
  Base Class for Anomaly Detection
"""

import abc
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AnomalyBase(BaseEstimator):
  """Anomaly detection base class
  """

  def __init__(self, norm='robust'):
    """Constructor
    """
    self.print_tag = type(self).__name__ # print Class name
    self.is_fitted = False # True if the model is already fitted
    self._features = None # User provided input data, reformatted into numpy array
    self._targets = None # User provided output data, reformatted into numpy array, the purpose of this data is depended on the algorithm
    self._norm = norm # type of normalization, either 'robust' or 'standard (z-score)'
    if norm is None:
      logger.info('Use standard scalar (z-score) to transform input data.')
      self._scalar = StandardScaler()
    elif norm.lower() == 'robust':
      logger.info('Use robust scalar to transform input data.')
      self._scalar = RobustScaler()
    else:
      logger.warning('Unrecognized value for param "norm", using default "RobustScalar"')
      self._scalar = RobustScaler()
    self._meta = {} # dictionary to store algorithm generated meta data
    self._xindex = None # store index for provided input data
    self._yindex = None # store index for provided output data
    self._xcolumns = None # store column names for provided x data
    self._ycolumns = None # store columns names for provided y data

  def reset(self):
    """reset
    """
    self.is_fitted = False
    self._features = None
    self._targets = None
    self._norm = 'robust'
    self._scalar = RobustScaler()
    self._meta = {}

  def get_params(self):
    """
    Get parameters for this estimator.

    Returns
    -------
    params : dict
        Parameter names mapped to their values.
    """
    params = super().get_params()
    return params

  def set_params(self, **params):
    """Set the parameters of this estimator.

    Parameters
    ----------
    **params : dict
        Estimator parameters.

    Returns
    -------
    self : estimator instance
        Estimator instance.
    """
    super().set_params(**params)


  def fit(self, X, y=None):
    """perform fitting

    Args:
        X (array-like): (n_samples, n_features)
        y (array-like, optional): (n_samples, n_features). Defaults to None.
    """
    logger.info('Train model.')
    self.is_fitted = True
    self._features, self._xindex, self._xcolumns = self.check_data(X)
    if y is not None:
      self._targets, self._yindex, self._ycolumns = self.check_data(y)
    X_transform = self._scalar.fit_transform(self._features)
    self._fit(X_transform, y)


  def evaluate(self, X):
    """perform evaluation

    Args:
        X (array-like): (n_samples, n_features)
    """
    logger.info('Perform model forecast.')
    X, index, columns = self.check_data(X)
    assert columns == self._xcolumns, 'Evaluated data should have the same number of columns!'
    X_transform = self._scalar.transform(X)
    self._xindex = np.hstack((self._xindex, index))
    self._features = np.vstack((self._features, X))
    self._evaluate(X_transform)

  def plot(self):
    """plot data
    """


  #################################################################################
  # To be implemented in subclasses

  @abc.abstractmethod
  def get_anomalies(self):
    """get the anomalies
    """

  @abc.abstractmethod
  def _fit(self, X, y=None):
    """perform fitting

    Args:
        X (array-like): (n_samples, n_features)
        y (array-like, optional): (n_samples, n_features). Defaults to None.
    """


  @abc.abstractmethod
  def _evaluate(self, X):
    """perform evaluation

    Args:
        X (array-like): (n_samples, n_features)
    """

  @staticmethod
  def check_data(data):
    """Check the format of data

    Args:
        data (_type_): list, numpy.ndarray or pandas.DataFrame
    """
    index = None
    columns = None
    data_ = None
    if isinstance(data, list):
      data_ = np.atleast_1d(data)
      if len(data_.shape) == 1:
        data_ = data_.reshape(data_.shape[0], 1)
      index = np.arange(data_.shape[0])
      columns = np.arange(data_.shape[1])
    elif isinstance(data, np.ndarray):
      data_ = data
      if len(data_.shape) == 1:
        data_ = data_.reshape(data_.shape[0], 1)
      index = np.arange(data_.shape[0])
      columns = np.arange(data_.shape[1])
    elif isinstance(data, pd.Series):
      data_ = data.to_numpy().reshape(len(data),1)
      index = data.index
      columns = [data.name] if data.name is not None else [0]
    elif isinstance(data, pd.DataFrame):
      data_ = data.to_numpy()
      index = data.index
      columns = data.columns
    else:
      raise IOError(f'The data with type {type(data)} cannot be accepted, please try to provide data with type of "list, numpy.array, pandas.Series or pandas.DataFrame"!')

    return data_, index, columns
