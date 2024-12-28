#!/usr/bin/env python

class BiasInitializer:
  """Base class for bias initialization."""
  def initialize_params(self):
    """Initialize bias vector.

    Returns
    -------
    np.array
      Initialized bias.
    """
    raise NotImplementedError()
