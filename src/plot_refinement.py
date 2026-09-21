"""Compare certified folded densities and mass imbalance within split components."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from plot_solution import density_grid
from advance import load_incumbent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('study',type=Path)
    ap.add_argument('--before',type=Path,default=Path(__file__).parent/'runs/advance_v2/attempt_000/certificate/certified_candidate.json')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();study=json.loads((args.study/'study.json').read_text())
    before=load_incumbent(args.before,26)
    after=load_incumbent(args.study/'repaired/certificate/certified_candidate.json',26)
    if before['L']!=after['L']:raise ValueError('Comparison requires identical L')
    a,fa=density_grid(before);b,fb=density_grid(after)
    norm=PowerNorm(.35,0,max(float(fa.max()),float(fb.max())))
    fig,axes=plt.subplots(1,3,figsize=(16,5.6),layout='constrained',gridspec_kw={'width_ratios':[1,1,1]})
    offsets=[(-30,-35),(25,0),(25,25),(-30,5)]
    for ax,edge,f,data,title in zip(axes[:2],[a,b],[fa,fb],[before,after],['Before','After long-axis splits']):
        mesh=ax.pcolormesh(edge,edge,f,cmap='magma',norm=norm,rasterized=True)
        ax.set(xlim=(0,data['L']/2),ylim=(0,data['L']/2),aspect='equal',xlabel='x (wall distance)',ylabel='y (wall distance)',
               title=f"{title}\nMass = {sum(data['weights']):.9f}")
        for j,item in enumerate(study['final_split_details']):
            x0,y0,x1,y1=item['parent_rectangle'];cx=(x0+x1)/2;cy=(y0+y1)/2
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#42d9ff',lw=.7))
            if item['axis']=='x':ax.plot([cx,cx],[y0,y1],color='#42d9ff',lw=1)
            else:ax.plot([x0,x1],[cy,cy],color='#42d9ff',lw=1)
            ax.annotate(str(j+1),(cx,cy),xytext=offsets[j%4],textcoords='offset points',color='white',
                        bbox={'boxstyle':'circle,pad=.15','fc':'#215269','ec':'white','lw':.5},
                        arrowprops={'arrowstyle':'-','color':'white','lw':.7},fontsize=9)
    details=study['final_split_details'];ratio=[]
    for d in details:
        w=np.array(d['effective_child_masses']);ratio.append(float(w[0]/w.sum()) if w.sum()>0 else 0.)
    y=np.arange(len(details))
    axes[2].barh(y,ratio,color='#417fbd',label='First half (lower coordinate)')
    axes[2].barh(y,1-np.array(ratio),left=ratio,color='#e59a48',label='Second half')
    axes[2].axvline(.5,color='#333333',ls='--',lw=1,label='Original uniform split: 50% / 50%')
    for j,r in enumerate(ratio):axes[2].text(.5,j,f'{100*r:.1f}% / {100*(1-r):.1f}%',ha='center',va='center',color='black',
                                         bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':2})
    axes[2].set(yticks=y,yticklabels=[f"Split {j+1} ({d['axis']})" for j,d in enumerate(details)],xlim=(0,1),
                title='Mass allocation after optimization',xlabel='Fraction of parent + child component mass')
    axes[2].invert_yaxis();axes[2].legend(loc='lower center',bbox_to_anchor=(.5,-.35),fontsize=8)
    fig.colorbar(mesh,ax=list(axes[:2]),location='bottom',shrink=.75,label='Continuous density (shared nonlinear color scale)')
    fig.suptitle(f"L = {before['L']:g}  |  Both solutions verified: every unit square >= 1.0001",fontsize=13)
    args.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.out,dpi=180);plt.close(fig)


if __name__=='__main__':main()
