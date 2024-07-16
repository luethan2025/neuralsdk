#!/usr/bin/env python

class Scheduler:
  """Base class for learning rate scheduler."""
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
