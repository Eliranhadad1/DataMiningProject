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
    i = 0
    sumTop = 0
    for x in x:
        sumTop = sumTop + (x - x_avg)*((y[i]) - y_avg)
        i++
# here loops end
        
    
    return sumTop/(len(x)-1)