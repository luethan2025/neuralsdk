#!/usr/bin/env python

import numpy as np

from ..base import Module, Parameter
from ..params.weights import Xavier
from ..params.bias import Zero

class LazyDense(Module):
  """NumPy implementation of the LazyDense Layer.

  Parameters
  ----------
  out_dim : int
    Length of output dimensions.
  weight_initializer : WeightInitializer
    Weight initialization method (defaults to Xavier).
  bias_initializer : BiasInitializer
    Bias initialization method (defaults to Zero).

  Notes:
  ------
  Lazy initialization of the in_dim argument. The in_dim argument is infered
  from the initial forward pass.
  """
  def __init__(
      self, out_dim, weight_initializer=Xavier, bias_initializer=Zero):
    self.initial_forward_pass = True
    self.out_dim = out_dim
    self.weight_initializer = weight_initializer
    self.bias_initializer = bias_initializer

  def forward(self, x):
    """Forward propagation through LazyDense.

    Parameters
    ----------
    x : np.array
      Input for this layer.
  
    Returns
    -------
    np.array
      Output of this layer.
    """
    if self.initial_forward_pass:
      in_dim, _ = x.shape
      W = self.weight_initializer(in_dim, self.out_dim).initialize_params()
      b = self.bias_initializer(self.out_dim).initialize_params()
      self.trainable_parameters = [Parameter(W), Parameter(b)]
      self.initial_forward_pass = False
    self.x = x
    W, b = self.trainable_parameters
    return np.tensordot(W.value, x, axes=[1, 1]).T + b.value

  def backward(self, grad):
    """Backward propagation for LazyDense.

    Parameters
    ----------
    grad : np.array
      Gradient (Loss w.r.t. data) flowing backwards from the next layer. Should
      have dimensions (batch, dim).
  
    Returns
    -------
    np.array
      Gradients for the inputs to this module. Should have dimensions
      (batch, dim).
    """
    W, b = self.trainable_parameters
    batch = self.x.shape[0]
    W.grad = np.matmul(grad.T, self.x) / batch
    b.grad = np.sum(grad, axis=0) / batch
    dx = np.matmul(grad, W.value)
    return dx
