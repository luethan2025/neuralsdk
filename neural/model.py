#!/usr/bin/env python

import numpy as np
from tqdm import tqdm

from .nn.base import Module
from .nn.modules import SoftmaxCrossEntropy
from .optim import supported_optimizers
from .optim.lr_scheduler import supported_lr_schedulers
from .optim.lr_scheduler import ConstantLR

def categorical_cross_entropy(pred, labels, epsilon=1e-10):
  """Cross entropy loss function.

  Parameters
  ----------
  pred : np.array
    Softmax label predictions. Should have shape (dim, num_classes).
  labels : np.array
    One-hot true labels. Should have shape (dim, num_classes).
  epsilon : float
    Small constant to add to the log term of cross entropy to help with
    numerical stability (defaults to 1e-10).

  Returns
  -------
  float
    Mean cross entropy loss in this batch.
  """
  return np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))

def categorical_accuracy(pred, labels):
  """Accuracy statistic.

  Parameters
  ----------
  pred : np.array
    Softmax label predictions. Should have shape (dim, num_classes).
  labels : np.array
    One-hot true labels. Should have shape (dim, num_classes).

  Returns
  -------
  float
    Mean accuracy in this batch.
  """
  return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))

def instantiate_loss(loss):
  """Instantiate loss function.

  Parameters
  ----------
  loss : Module
    Final output activation and loss function.

  Return
  ------
  Module
    Instantiated loss function.
  """
  if loss in [SoftmaxCrossEntropy]:
    return loss()
  elif isinstance(loss, SoftmaxCrossEntropy):
    return loss

def instantiate_optimizer(optimizer):
  """Instantiate optimization policy.

  Parameters
  ----------
  optimizer : Optimizer
    Optimization policy.

  Return
  ------
  Optimizer
    Instantiated optimizer.
  """
  if optimizer in supported_optimizers:
    return optimizer()
  elif any([
    isinstance(optimizer, optim) 
      for optim in supported_optimizers
   ]):
    return optimizer

def instantiate_lr_scheduler(lr_scheduler):
  """Instantiate learning rate scheduler.

  Parameters
  ----------
  lr_scheduler : Scheduler
    Learning rate scheduler.

  Return
  ------
  Scheduler
    Instantiated learning rate scheduler.
  """
  if lr_scheduler is None:
    return ConstantLR()
  elif lr_scheduler in supported_lr_schedulers:
    return lr_scheduler()
  elif any([
    isinstance(lr_scheduler, scheduler)
      for scheduler in supported_lr_schedulers
  ]):
    return lr_scheduler


class Sequential:
  """Sequential neural network model.

  Parameters
  ----------
  modules : Module[]
    List of modules; used to grab trainable weights.
  loss : Module
    Final output activation and loss function.
  optimizer : Optimizer
    Optimization policy.
  lr_scheduler : Scheduler
    Learning rate scheduler.
  """
  def __init__(self, modules, loss=None, optimizer=None, lr_scheduler=None):
    for module in modules:
      assert(isinstance(module, Module))
    assert(loss is not None)
    assert(False if optimizer is None and lr_scheduler is not None else True)
    assert(optimizer is not None)
    assert(optimizer in supported_optimizers or \
      any([
        isinstance(optimizer, optim) 
          for optim in supported_optimizers
      ])
    )
    assert(lr_scheduler in supported_lr_schedulers or \
      any([
        isinstance(lr_scheduler, scheduler)
          for scheduler in supported_lr_schedulers
        ])
    )

    self.modules = modules
    self.loss = instantiate_loss(loss)

    self.params = []
    for module in modules:
      self.params += module.trainable_parameters

    self.optimizer = instantiate_optimizer(optimizer)
    self.optimizer.initialize_params(self.params)

    self.lr_scheduler = instantiate_lr_scheduler(lr_scheduler)
    self.lr_scheduler.set_optimizer(self.optimizer)

  def forward(self, X):
    """Model forward pass.

    Parameters
    ----------
    X : np.array
      Input data.

    Returns
    -------
    np.array
      Batch predictions; should have shape (batch, num_classes).
    """
    for module in self.modules:
      X = module.forward(X)
    return self.loss.forward(X)

  def backward(self, y):
    """Model backwards pass.

    Parameters
    ----------
    y : np.array
      True labels.
    """
    grad = self.loss.backward(y)
    for module in reversed(self.modules):
      grad = module.backward(grad)

  def train(self, dataset):
    """Fit model on dataset for a single epoch.

    Parameters
    ----------
    dataset : Dataset
      Training dataset with batches already split.

    Returns
    -------
    (float, float)
      [0] Mean train loss during this epoch.
      [1] Mean train accuracy during this epoch.
    """
    losses = np.zeros(shape=dataset.size)
    accuracy = np.zeros(shape=dataset.size)
    with tqdm(
        total=dataset.size,
        postfix={"loss": 0, "accuracy": 0}) as pbar:
      for i, batch in enumerate(dataset):
        X, y = batch
        pred = self.forward(X)
        self.backward(y)
        self.optimizer.apply_gradients(self.params)

        losses[i] = categorical_cross_entropy(pred, y)
        accuracy[i] = categorical_accuracy(pred, y)
        pbar.update(1)
        pbar.set_postfix(loss=losses[i], accuracy=accuracy[i])
    self.lr_scheduler.step()
    return np.mean(losses), np.mean(accuracy)
  
  def test(self, dataset):
    """Compute test/validation loss for dataset.

    Parameters
    ----------
    dataset : Dataset
      Validation dataset with batches already split.

    Returns
    -------
    (float, float)
      [0] Mean test loss.
      [1] Test accuracy.
    """
    pred = self.forward(dataset.X)
    loss = categorical_cross_entropy(pred, dataset.y)
    acc = categorical_accuracy(pred, dataset.y)
    return loss, acc
