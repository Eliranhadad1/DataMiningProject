# -*- coding: utf-8 -*-
"""
This function calculate r(x,y)
@author: Eliran
"""
import numpy as np

def r_function(x,y):
    cov = cov(x,y)
    sx = s_x2(x)
    sy = s_x2(y)
    
    return cov/(sx*sy)
    