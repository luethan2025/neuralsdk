#!/usr/bin/env python

from .base import Scheduler

class ConstantLR(Scheduler):
  """Maintain the original learning rate."""
  def __init__(self):
    super().__init__()

  def step(self):
    """Apply learning rate update policy."""
  
class StepLR(Scheduler):
  """Decay the learning rate by gamma each step_size epoch.

  Parameters
  ----------
  step_size : int
    Period of learning rate decay.
  gamma : float 
    Learning rate decay parameter (defaults to 0.1).
  """
  def __init__(self, step_size, gamma=0.1):
    self.step_size = step_size
    self.gamma = gamma
    self.last_epoch = 0
    super().__init__()

  def step(self):
    """Apply learning rate update policy."""
    self.last_epoch = self.last_epoch + 1
    if not (self.last_epoch % self.step_size):
      self.last_lr = self.optimizer.get_lr()
      self.optimizer.adjust_lr(self.gamma)

      assert(self.optimizer.get_lr() == self.last_lr * self.gamma)

class MultiStepLR(Scheduler):
  """Decay the learning rate by gamma when the number of epoch reaches a
  specific milestone.

  Parameters
  ----------
  milestones : int[]
    List of epoch indices.
  gamma : float 
    Learning rate decay parameter (defaults to 0.1).
  """
  def __init__(self, milestones, gamma=0.1):
    self.milestones = milestones
    self.gamma = gamma
    self.last_epoch = 0
    super().__init__()

  def step(self):
    """Apply learning rate update policy."""
    self.last_epoch = self.last_epoch + 1
    if self.last_epoch in self.milestones:
      self.last_lr = self.optimizer.get_lr()
      self.optimizer.adjust_lr(self.gamma)

      assert(self.optimizer.get_lr() == self.last_lr * self.gamma)

class ChainedScheduler(Scheduler):
  """Chain multiple learning rate schedulers together.

  Parameters
  ----------
  schedulers : Scheduler[]
    List of learning rate schedulers.
  """
  def __init__(self, schedulers):
    for lr_scheduler in schedulers:
      assert(any([
        isinstance(lr_scheduler, scheduler)
          for scheduler in supported_lr_schedulers
      ]))

    self.schedulers = schedulers
    
  def set_optimizer(self, optimizer):
    """Change the optimizer policy of the learning rate scheduler points to.

    Parameters
    ----------
    optimizer : Optimizer
      Optimizer policy.
    """
    for scheduler in self.schedulers:
      scheduler.set_optimizer(optimizer)

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

supported_lr_schedulers = [ConstantLR, StepLR, MultiStepLR, ChainedScheduler]
