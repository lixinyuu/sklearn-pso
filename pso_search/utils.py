# -*- coding: utf-8 -*-
from __future__ import division
import os
import warnings

import numpy as np

class BaseMapFunction:
    """
    The base class used to generate a map function from PSO searched 
    values to real values.
    """
    def __init__(self, func, low, high):
        self.map_func = func
        self.low = low
        self.high = high

class Bound(BaseMapFunction):
    """
    The class used to generate the bound condition in 
    the PSO optimation
    """
    def __init__(self, a, b):
        assert a != b
        self.logbase = 1
        self.low = np.min([a, b])
        self.high = np.max([a, b])
        self.map_func = lambda x: x

class LogSpace(BaseMapFunction):
    """
    The class used to define a log space for PSO to search
    If we are going to find a paramers between 1e-9 to 1e9,
    it is easily to guess that most of the initial guess will 
    be located between 1e8 to 1e9, which prevent the PSO 
    algorithms to search the space between 1e-9 to 1e8. In this
    implementation, we use this class to make the parameters that
    are searched in PSO is -9 to 9, and then use logbase to retrive 
    the real value.
    """
    def __init__(self, logbase, a, b):
        assert a != b
        self.logbase = logbase
        self.low = np.min([a, b])
        self.high = np.max([a, b])
        self.map_func =  lambda x: self.logbase ** x

class IntBound(BaseMapFunction):
    """
    The class used to generate the bound condition in 
    the PSO optimation
    """
    def __init__(self, a, b):
        assert a != b
        self.logbase = 1
        self.low = np.min([a, b])
        self.high = np.max([a, b])
        self.map_func = lambda x: np.around(x).astype(int)