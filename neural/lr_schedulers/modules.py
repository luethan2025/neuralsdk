#!/usr/bin/env python

from .base import Scheduler

class ConstantLR(Scheduler):
  """Maintain the original learning rate.

  Parameters
  ----------
  optimizer : Optimizer
    Optimizer policy.
  """
  def __init__(self, optimizer):
    self.optimizer = optimizer

  def step(self):
    """Apply learning rate update policy."""
  