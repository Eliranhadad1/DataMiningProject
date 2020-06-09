
from scipy.stats import spearmanr
def zv(Di,deltaT):
    T=len(deltaT)
    sigma=0;
    for Dj in deltaT:
        [c,p]=spearmanr(Di,Dj)
        sigma+=c
    return (1/ T)*sigma


