# -*- coding: utf-8 -*-
"""
This function calculate covariance

I assumed that the number of "x" is equal to the number of "y"

* x and y are lists!

@author: Eliran
"""


import numpy as np

def cov(x,y):
    x_avg = sum(x)/(len(x))
    y_avg = sum(y)/(len(y)
    
# sumTop is the top part of the formula
    
    for point in x:
        sumTop = sumTop + (x[i] - x_avg)*((y[i]) - y_avg)
        
# here loops end
        
    
    return sumTop/(len(x)-1)