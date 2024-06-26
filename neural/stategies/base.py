#!/usr/bin/env python

class Strategy:
  """Base class for weight and bias initialization strategy.

  Parameters
  ----------
  in_dim : int
    Length of input dimensions.
  out_dim : int
    Length of output dimensions.
  """
  def __init__(self, in_dim, out_dim):
    self.in_dim = in_dim
    self.out_dim = out_dim

  def initialize_weights_and_bias(self):
    """Use initialization strategy

    Returns
    ------
    (np.array, np.array)
      [0] Initialized weight.
      [1] Initialized bias.
    """
    raise NotImplementedError()
