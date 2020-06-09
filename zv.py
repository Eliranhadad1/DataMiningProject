
def zv(Di,deltaT):
    T=deltaT.size()
    sigma=0;
    for Dj in deltaT:
        sigma+=R(Di,Dj)
    return (1/ T)*sigma

def R(Di,Dj):
    d=[]
    m=len(Di)
    for jj in range(m):
        d=Di[jj]-Dj[jj]
    
    d2=[dvalue*dvalue for dvalue in d]
    d2sum=sum(d2,0)
    return (1-(6*d2sum)/(m*(m**2-1)));
