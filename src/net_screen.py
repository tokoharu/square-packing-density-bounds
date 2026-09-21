"""Floating point branch-and-bound screening; NOT a rigorous certificate."""
import numpy as np
from numba import njit
from fast_geometry import overlap

@njit(cache=True)
def length_bounds(z,center,other,rad,other_rad,lo,hi,A,k1,k2):
    # Slice endpoints: other +/- A/den + k*(z-center).
    m1=other+k1*(z-center);m2=other+k2*(z-center)
    e1=other_rad+abs(k1)*rad;e2=other_rad+abs(k2)*rad
    low=max(lo,m1-A[0]+e1,m2-A[1]+e2)
    high=min(hi,m1+A[0]-e1,m2+A[1]-e2)
    lower=max(0.,high-low)
    low=max(lo,m1-A[0]-e1,m2-A[1]-e2)
    high=min(hi,m1+A[0]+e1,m2+A[1]+e2)
    upper=max(0.,high-low)
    return lower,upper

@njit(cache=True)
def bound(cx,cy,dx,dy,c,s,B,rects,rho):
    f=0.;fxlo=0.;fxhi=0.;fylo=0.;fyhi=0.
    ex=B*(c+s)/2
    for k in range(len(rho)):
        a,b,d,e=rects[k]
        if d<cx-dx-ex or a>cx+dx+ex or e<cy-dy-ex or b>cy+dy+ex:continue
        f+=rho[k]*overlap(rects[k],cx,cy,c,s,B)
        if s==0:continue
        A=np.array([B/(2*s),B/(2*c)])
        l0,u0=length_bounds(a,cx,cy,dx,dy,b,e,A,-c/s,s/c)
        l1,u1=length_bounds(d,cx,cy,dx,dy,b,e,A,-c/s,s/c)
        fxlo+=rho[k]*(l0-u1);fxhi+=rho[k]*(u0-l1)
        A=np.array([B/(2*c),B/(2*s)])
        l0,u0=length_bounds(b,cy,cx,dy,dx,a,d,A,-s/c,c/s)
        l1,u1=length_bounds(e,cy,cx,dy,dx,a,d,A,-s/c,c/s)
        fylo+=rho[k]*(l0-u1);fyhi+=rho[k]*(u0-l1)
    if s==0:
        low=1e100
        for sx in [-1,1]:
            for sy in [-1,1]:
                val=0.
                for k in range(len(rho)):val+=rho[k]*overlap(rects[k],cx+sx*dx,cy+sy*dy,c,s,B)
                low=min(low,val)
        # Bilinear only if the cell crosses no axis-aligned event line.
        return -1.,f,0.,0.
    gx=max(abs(fxlo),abs(fxhi));gy=max(abs(fylo),abs(fyhi))
    return f-dx*gx-dy*gy,f,dx*gx,dy*gy

@njit(cache=True)
def verify_angle(c,s,B,L,rects,rho,budget=3000000):
    # Stack contains physical center boxes in the upper-right quadrant.
    stack=np.empty((2000,4));top=1;e=(L-B*(c+s))/2
    stack[0]=np.array([L/2+e/2,L/2+e/2,e/2,e/2])
    visited=0;leaves=0;lowest=1e100;witness=np.zeros(3);minlower=1e100
    while top:
        top-=1;cx,cy,dx,dy=stack[top];visited+=1
        low,value,rx,ry=bound(cx,cy,dx,dy,c,s,B,rects,rho)
        if value<lowest:lowest=value;witness=np.array([cx,cy,value])
        # A conservative rejection band overlaps the acceptance threshold, so
        # cells on a level set cannot recurse indefinitely due to rounding.
        if value<1+1e-7:return 0,visited,leaves,lowest,witness,top,minlower
        if low>=1+1e-9:leaves+=1;minlower=min(minlower,low);continue
        if visited>=budget:return -1,visited,leaves,lowest,witness,top,minlower
        if top>=1900 or max(dx,dy)<1e-12:return -1,visited,leaves,lowest,witness,top,minlower
        if rx>=ry:
            dx/=2
            stack[top]=np.array([cx-dx,cy,dx,dy]);stack[top+1]=np.array([cx+dx,cy,dx,dy])
        else:
            dy/=2
            stack[top]=np.array([cx,cy-dy,dx,dy]);stack[top+1]=np.array([cx,cy+dy,dx,dy])
        top+=2
    return 1,visited,leaves,lowest,witness,top,minlower
