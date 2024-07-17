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
    Learning rate decay parameter (defaults to 0.1).
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

class MultiStepLR(Scheduler):
  """Decay the learning rate by gamma when the number of epoch reaches a
  specific milestone.

  Parameters
  ----------
  optimizer : Optimizer
    Optimizer policy.
  milestones : int[]
    List of epoch indices.
  gamma : float 
    Learning rate decay parameter (defaults to 0.1).
  """
  def __init__(self, optimizer, milestones, gamma=0.1):
    self.milestones = milestones
    self.gamma = gamma
    self.last_epoch = 0
    super().__init__(optimizer)

  def step(self):
    """Apply learning rate update policy."""
    self.last_epoch = self.last_epoch + 1
    if self.last_epoch in self.milestones:
      self.optimizer.adjust_lr(self.gamma)

class ChainedScheduler(Scheduler):
  """Chain multiple learning rate schedulers together.

  Parameters
  ----------
  schedulers : Scheduler[]
    List of learning rate schedulers.
  """
  def __init__(self, schedulers):
    self.schedulers = schedulers

  def get_schedulers(self):
    """Return the list of learning rate schedulers.
  
    Return
    ------
    Scheduler[]
      List of learning rate schedulers.
    """
    return self.schedulers

  def step(self):
    """Apply learning rate update policy."""
    for s in self.schedulers:
      s.step()
