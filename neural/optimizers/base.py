#!/usr/bin/env python

class Optimizer:
  """Base class for optimization policy."""
  def initialize_params(self, params):
    """Initialize optimizer state.

    Parameters
    ----------
    params : Parameter[]
      List of parameters to initialize state for.
    """

  def adjust_lr(self, gamma):
    """Adjust learning rate by a fixed rate.

    Parameters
    ----------
    gamma : float
      Learning rate decay parameter.
    """
    self.lr *= gamma

  def apply_gradients(self, params):
    """Apply gradients to parameters.

    Parameters
    ----------
    params : Parameter[]
      List of parameters that the gradients correspond to.
    """
    raise NotImplementedError()
