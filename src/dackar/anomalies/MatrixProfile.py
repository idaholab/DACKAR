import numpy as np

from .AnomalyBase import AnomalyBase

import logging

logger = logging.getLogger(__name__)


class MatrixProfile(AnomalyBase):
  """_summary_

  Args:
      AnomalyBase (_type_): _description_
  """


  def __init__(self, norm='robust'):
    """Constructor
    """
    super().__init__(norm=norm)



  def _fit(self, X, y=None):
    """perform fitting

    Args:
        X (array-like): (n_samples, n_features)
        y (array-like, optional): ignored, (n_samples, n_features). Defaults to None.
    """



  def _predict(self, X):
    """perform prediction

    Args:
        X (array-like): (n_samples, n_features)
    """
