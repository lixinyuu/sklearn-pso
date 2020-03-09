# -*- coding: utf-8 -*-
from __future__ import division
import os
import warnings

import numpy as np

class Bound:
    """
    The class used to generate the bound condition in 
    the PSO optimation
    """
    def __init__(self, a, b):
        assert a != b
        self.logbase = 1
        self.low = np.min([a, b])
        self.high = np.max([a, b])

class LogSpace:
    def __init__(self, logbase, a, b):
        assert a != b
        self.logbase = logbase
        self.low = np.min([a, b])
        self.high = np.max([a, b])