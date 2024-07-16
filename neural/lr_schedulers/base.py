#!/usr/bin/env python

class Scheduler:
  """Base class for learning rate scheduler.
  
  Parameters
  ----------
  optimizer : Optimizer
    Optimizer policy.
  """
  def __init__(self, optimizer):
    self.optimizer = optimizer

  def set_optimizer(self, optimizer):
    """Change the optimizer policy of the learning rate scheduler points to.

    Parameters
    ----------
    optimizer : Optimizer
      Optimizer policy.
    """
    self.optimizer = optimizer

  def step():
    """Apply learning rate update policy."""
    raise NotImplementedError()
