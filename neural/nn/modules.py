#!/usr/bin/env python

import numpy as np

from .base import Module, Parameter

class Dense(Module):
  """NumPy implementation of the Dense Layer.

  Parameters
  ----------
  in_dim : int
    Length of input dimensions.
  out_dim : int
    Length of output dimensions.
  initialization_technique : Techique
    Weight and bias initialization technique.
  """
  def __init__(self, in_dim, out_dim, initialization_technique=None):
    W, b = initialization_technique(in_dim, out_dim).initialize_weights_and_bias()
    self.trainable_parameters = [Parameter(W), Parameter(b)]

  def forward(self, x):
    """Forward propagation through Dense.

    Parameters
    ----------
    x : np.array
      Input for this layer.
  
    Returns
    -------
    np.array
      Output of this layer.
    """
    self.x = x
    W, b = self.trainable_parameters
    return np.tensordot(W.value, x, axes=[1, 1]).T + b.value

  def backward(self, grad):
    """Backward propagation for Dense.

    Parameters
    ----------
    grad : np.array
      Gradient (Loss w.r.t. data) flowing backwards from the next layer. Should
      have dimensions (batch, dim).
  
    Returns
    -------
    np.array
      Gradients for the inputs to this module. Should have dimensions
      (batch, dim).
    """
    W, b = self.trainable_parameters
    batch = self.x.shape[0]
    W.grad = np.matmul(grad.T, self.x) / batch
    b.grad = np.sum(grad, axis=0) / batch
    dx = np.matmul(grad, W.value)
    return dx

class Sigmoid(Module):
  """NumPy implementation the Sigmoid Activation."""
  def __init__(self):
    super().__init__()
  
  def forward(self, x):
    """Forward propagation through Sigmoid.

    Parameters
    ----------
    x : np.array
      Input for this layer.
  
    Returns
    -------
    np.array
      Output of this layer.
    """
    self.x = x
    fx = 1 / (1 + np.exp(-x))
    self.fx = fx
    return fx

  def backward(self, grad):
    """
    Backward propogation for Sigmoid.

    Parameters
    ----------
    grad : np.array
      Gradient (Loss w.r.t. data) flowing backwards from the next layer,
      dL/dx_k. Should have dimensions (batch, dim).

    Returns
    -------
    np.array
      Gradients for the inputs to this layer, dL/dx_{k-1}. Should
      have dimensions (batch, dim).
    """
    dLdx = grad * (self.fx * (1 - self.fx))
    return dLdx

class Tanh(Module):
  """NumPy implementation the Tanh Activation (Hyperbolic Tangent)."""
  def __init__(self):
    super().__init__()

  def forward(self, x):
    """Forward propagation through Tanh.

    Parameters
    ----------
    x : np.array
      Input for this layer.
  
    Returns
    -------
    np.array
      Output of this layer.
    """
    self.x = x
    fx = np.divide(
      np.exp(x) - np.exp(-x), np.exp(x) + np.exp(-x))
    self.fx = fx
    return fx

  def backward(self, grad):
    """
    Backward propogation for Tanh.

    Parameters
    ----------
    grad : np.array
      Gradient (Loss w.r.t. data) flowing backwards from the next layer,
      dL/dx_k. Should have dimensions (batch, dim).

    Returns
    -------
    np.array
      Gradients for the inputs to this layer, dL/dx_{k-1}. Should
      have dimensions (batch, dim).
    """
    dLdx = grad * (1 - np.square(self.fx))
    return dLdx

class ReLU(Module):
  """NumPy implementation the ReLU Activation (Rectified Linear Unit)."""
  def __init__(self):
    super().__init__()

  def forward(self, x):
    """Forward propagation through ReLU.

    Parameters
    ----------
    x : np.array
      Input for this layer.
  
    Returns
    -------
    np.array
      Output of this layer.
    """
    self.x = x
    fx = np.copy(x)
    fx[x <= 0] = 0
    return fx

  def backward(self, grad):
    """
    Backward propogation for ReLU.

    Parameters
    ----------
    grad : np.array
      Gradient (Loss w.r.t. data) flowing backwards from the next layer,
      dL/dx_k. Should have dimensions (batch, dim).

    Returns
    -------
    np.array
      Gradients for the inputs to this layer, dL/dx_{k-1}. Should
      have dimensions (batch, dim).
    """
    dxdx = np.ones_like(self.x)
    dxdx[self.x <= 0] = 0
    dLdx = grad * dxdx
    return dLdx

class SoftmaxCrossEntropy(Module):
  """Softmax Cross Entropy fused output activation."""
  def __init__(self):
    super().__init__()

  def forward(self, logits):
    """Forward propagation through Softmax.

    Parameters
    ----------
    logits : np.array
      Softmax logits. Should have shape (batch, num_classes).

    Returns
    -------
    np.array
      Predictions for this batch. Should have shape (batch, num_classes).
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    self.y_pred = np.divide(
        exp_logits, np.sum(exp_logits, axis=1, keepdims=True))
    return self.y_pred

  def backward(self, labels):
    """Backward propagation of the Softmax activation.

    Parameters
    ----------
    labels : np.array
      One-hot encoded labels. Should have shape (batch, num_classes).

    Returns
    -------
    np.array
      Initial backprop gradients.
    """
    return self.y_pred - labels
