#!/usr/bin/env python

from ..base import Module

class Flatten(Module):
  """NumPy implementation of the Flatten Layer.

  Notes:
  ------
  Flatten image into vector.
  """
  def __init__(self):
    super().__init__()

  def forward(self, x):
    """Forward propagation through Flatten.
    
    Parameters
    ----------
    x : np.array
      Input for this layer.
  
    Returns
    -------
    np.array
      Output of this layer.
    """
    self.shape = x.shape
    return x.reshape(x.shape[0], -1)

  def backward(self, grad):
    """
    Backward propogation for Flatten.

    Parameters
    ----------
    grad : np.array
      Gradient (Loss w.r.t. data) flowing backwards from the next layer,
      dL/dx_k. Should have dimensions (batch, dim).

    Returns
    -------
    np.array
      Gradients for the inputs to this layer, dL/dx_{k-1}. Should
      have dimensions (batch, dim).
    """
    return grad.reshape(self.shape)
