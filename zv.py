
def zv(Di,deltaT):
    T=deltaT.size()
    sigma=0;
    for Dj in deltaT:
        sigma+=R(Di,Dj)
    return (1/ T)*sigma

        
