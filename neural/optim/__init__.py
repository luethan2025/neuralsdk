#!/usr/bin/env python

from .optimizer import SGD, Adam

supported_optimizers = [SGD, Adam]
__all__ = [
  "supported_optimizers",
  "SGD", "Adam"
]
