#!/usr/bin/env python

from ..base import Module

class Flatten(Module):
  """Flatten image into vector."""
  def forward(self, x):
    """Forward propagation."""
    self.shape = x.shape
    return x.reshape(x.shape[0], -1)

  def backward(self, grad):
    """Backward propagation."""
    return grad.reshape(self.shape)
