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
    super().__init__(optimizer)

  def step(self):
    """Apply learning rate update policy."""
  
class StepLR(Scheduler):
  """Decay the learning rate by gamma each step_size epoch.

  Parameters
  ----------
  optimizer : Optimizer
    Optimizer policy.
  step_size : int
    Period of learning rate decay.
  gamma : float 
    Learning rate decay parameter.
  """
  def __init__(self, optimizer, step_size, gamma=0.1):
    self.step_size = step_size
    self.gamma = gamma
    self.last_epoch = 0
    super().__init__(optimizer)

  def step(self):
    """Apply learning rate update policy."""
    self.last_epoch = self.last_epoch + 1
    if not (self.last_epoch % self.step_size):
      self.optimizer.adjust_lr(self.gamma)
