# -*- coding: utf-8 -*-
"""
This function calculate s(x)^2
@author: Eliran
"""

import numpy as np
def s_x2(x):
    
    x_avg = sum(x)/(len(x))

# For loop, run from the first item till the last one (the first is included)
    for point in x:
        sum_square = sum_square + (x[i] - x_avg)**2
        

# here loops end
        

    
    return s_x2/(len(x)-1)