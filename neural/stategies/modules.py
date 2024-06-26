#!/usr/bin/env python

import numpy as np

from .base import Strategy

class Xavier(Strategy):
  """Base class for weight and bias initialization strategies.

  Parameters
  ----------
  in_dim : int
    Length of input dimensions.
  out_dim : int
    Length of output dimensions.
  """
  def __init__(self, in_dim, out_dim):
    super().__init__(in_dim, out_dim)

  def initialize_weights_and_bias(self):
    """Use Xavier initialization strategy

    Returns
    ------
    (np.array, np.array)
      [0] Initialized weight.
      [1] Initialized bias.
    """
    u = np.sqrt(6 / float(self.in_dim))
    W = np.random.uniform(low=-u, high=u, size=(self.out_dim, self.in_dim))
    b = np.zeros(shape=(self.out_dim))

    return W, b
