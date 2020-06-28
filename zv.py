
from scipy.stats import spearmanr
import pandas
def zv(Di,deltaT):
    
    T=len(deltaT)
    if T==0:
        return 
    
    sigma=0;
    for Dj in deltaT:
        [c,p]=spearmanr(Di,Dj)
        sigma+=c
    return (1/ T)*sigma

x=zv([68,53,70],[[20,15,80]])

# [68,53,70]--->[2,3,1]
# [20,15,80]--->[2,3,1]

# 1- 0 = 1