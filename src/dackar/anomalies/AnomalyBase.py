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
import logging

logger = logging.getLogger(__name__)

class AnomalyBase(BaseEstimator):
  """Anomaly detection base class
  """

  def __init__(self, norm='robust'):
    """Constructor
    """
    self.print_tag = type(self).__name__
    self.is_fitted = False
    self._features = None
    self._targets = None
    self._norm = norm
    if norm is None:
      logger.info('Use standard scalar (z-score) to transform input data.')
      self._scalar = StandardScaler()
    elif norm.lower() == 'robust':
      logger.info('Use robust scalar to transform input data.')
      self._scalar = RobustScaler()
    else:
      logger.warning('Unrecognized value for param "norm", using default "RobustScalar"')
      self._scalar = RobustScaler()
    self._meta = {}

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
    if not isinstance(X, pd.DataFrame):
      raise IOError(f'Pandas.DataFrame is required, but get {type(X)}!')
    if y is not None and not isinstance(y, pd.DataFrame):
      raise IOError(f'Pandas.DataFrame is required, but get {type(y)}!')
    self._features = X
    self._targets = y
    X_transform = self._scalar.fit_transform(X)
    X_transform = pd.DataFrame(X_transform, columns=X.columns)
    fit_obj = self._fit(X_transform, y)
    return fit_obj

  def evaluate(self, X):
    """perform evaluation

    Args:
        X (array-like): (n_samples, n_features)
    """
    logger.info('Perform model forecast.')
    X_transform = self._scalar.fit_transform(X)
    X_transform = pd.DataFrame(X_transform, columns=X.columns)
    y_new = self.evaluate(X_transform)

    return y_new

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
