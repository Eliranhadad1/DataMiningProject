
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


