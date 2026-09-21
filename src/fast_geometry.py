"""Independent double-precision polygon clipping implementation, checked against GEOS."""
import numpy as np
from numba import njit

@njit(cache=True)
def overlap(rect,cx,cy,c,s,B):
    ext=B*(c+s)/2
    if rect[2]<=cx-ext or rect[0]>=cx+ext or rect[3]<=cy-ext or rect[1]>=cy+ext:return 0.
    dx=(rect[0]+rect[2])/2-cx;dy=(rect[1]+rect[3])/2-cy
    hx=(rect[2]-rect[0])/2;hy=(rect[3]-rect[1])/2
    if abs(c*dx+s*dy)+c*hx+s*hy<=B/2 and abs(-s*dx+c*dy)+s*hx+c*hy<=B/2:
        return (2*hx)*(2*hy)
    p=np.empty((12,2));q=np.empty((12,2));n=4
    for j in range(4):
        a=(-1 if j==0 or j==3 else 1)*B/2
        b=(-1 if j<2 else 1)*B/2
        p[j,0]=cx+c*a-s*b;p[j,1]=cy+s*a+c*b
    for side in range(4):
        axis=side%2;bound=rect[side];sign=1 if side<2 else -1
        count=0
        for j in range(n):
            k=(j+n-1)%n;dj=sign*(p[j,axis]-bound);dk=sign*(p[k,axis]-bound)
            if (dj>=0)!=(dk>=0):
                t=dk/(dk-dj)
                q[count,0]=p[k,0]+t*(p[j,0]-p[k,0]);q[count,1]=p[k,1]+t*(p[j,1]-p[k,1]);count+=1
            if dj>=0:q[count,0]=p[j,0];q[count,1]=p[j,1];count+=1
        if count<3:return 0.
        tmp=p;p=q;q=tmp;n=count
    area=0.
    for j in range(1,n-1):
        area+=(p[j,0]-p[0,0])*(p[j+1,1]-p[0,1])-(p[j,1]-p[0,1])*(p[j+1,0]-p[0,0])
    return abs(area)/2

@njit(cache=True)
def matrix(poses,L,B,full,areas,ids):
    out=np.zeros((len(poses),len(ids)))
    for i in range(len(poses)):
        theta=poses[i,2]*np.pi/4;c=np.cos(theta);s=np.sin(theta);e=(L-B*(c+s))/2
        cx=L/2+poses[i,0]*e;cy=L/2+poses[i,1]*e
        for j in range(len(ids)):
            for k in range(8):
                z=ids[j]*8+k
                out[i,j]+=overlap(full[z],cx,cy,c,s,B)/areas[z]/8
    return out

def fast_matrix(self,poses,selection=None):
    ids=np.arange(len(self.rects)) if selection is None else np.asarray(selection)
    return matrix(np.atleast_2d(poses),self.L,self.B,self.full,self.areas,ids)
