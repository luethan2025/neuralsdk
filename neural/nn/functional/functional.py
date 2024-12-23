#!/usr/bin/env python

import numpy as np

def sigmoid(x):
  """Functional version of Sigmoid Activation.

  Parameters
  ----------
  x : np.array
    Input data.

  Returns
  -------
  np.array
  """
  fx = 1 / (1 + np.exp(-x))
  return fx

def tanh(x):
  """Functional version of Tanh Activation (Hyperbolic Tangent).

  Parameters
  ----------
  x : np.array
    Input data.

  Returns
  -------
  np.array
  """
  fx = np.divide(
    np.exp(x) - np.exp(-x), np.exp(x) + np.exp(-x))
  return fx

def relu(x):
  """Functional version of ReLU Activation (Rectified Linear Unit).

  Parameters
  ----------
  x : np.array
    Input data.

  Returns
  -------
  np.array
  """
  fx = np.copy(x)
  fx[x <= 0] = 0
  return fx

def softmax_cross_entropy(logits):
  """Functional version of Softmax Cross Entropy.

  Parameters
  ----------
  logits : np.array
    Softmax logits.

  Returns
  -------
  np.array
  """
  exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
  y_pred = np.divide(
    exp_logits, np.sum(exp_logits, axis=1, keepdims=True))
  return y_pred
