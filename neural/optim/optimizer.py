#!/usr/bin/env python

import numpy as np

from .base import Optimizer

class SGD(Optimizer):
  """Stochastic Gradient Descent (SGD) optimizer.

  Parameters
  ----------
  lr : float
    Learning rate multiplier (defaults to 0.01).
  """
  def __init__(self, lr=0.01):
    self.lr = lr

  def initialize_params(self, params):
    """Initialize optimizer state.

    params : np.array[]
      List of parameters that will be used with this optimizer.
    """

  def apply_gradients(self, params):
    """Apply gradients to parameters.

    Parameters
    ----------
    params : Parameter[]
      List of parameters that the gradients correspond to.
    """
    for p in params:
      p.value -= self.lr * p.grad

class Adam(Optimizer):
  """Adam (Adaptive Moment) optimizer.

  Parameters
  ----------
  lr : float
    Learning rate multiplier (defaults to 0.001).
  beta1 : float
    Momentum decay parameter (defaults to 0.9).
  beta2 : float
    Variance decay parameter (defaults to 0.999).
  epsilon : float
    A small constant added to the denominator for numerical stability
    (defaults to 1e-7).
  """
  def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

  def initialize_params(self, params):
    """Initialize optimizer state.

    params : np.array[]
      List of parameters that will be used with this optimizer.
    """
    for p in params:
      p.m = np.zeros_like(p.value)
      p.v = np.zeros_like(p.value)

  def apply_gradients(self, params):
    """Apply gradients to parameters.

    Parameters
    ----------
    params : Variable[]
        List of parameters that the gradients correspond to.
    """
    for p in params:
      p.m = p.m * self.beta1 + p.grad * (1 - self.beta1)
      p.v = p.v * self.beta2 + np.square(p.grad) * (1 - self.beta2)
      update = (
          (p.m / (1 - self.beta1))
          / (np.sqrt(p.v / (1 - self.beta2)) + self.epsilon))
      p.value -= self.lr * update

supported_optimizers = [SGD, Adam]
