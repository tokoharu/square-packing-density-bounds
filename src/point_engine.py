"""Fixed-point D4 weight LP with incremental rows from the original point validator."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy import sparse
import highspy
from point_export import run_validator

VENDOR=Path(__file__).parent/'vendor/point_validator/verify.py'

def load_validator(path):
    spec=importlib.util.spec_from_file_location('original_validator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def orbit(x,y,L):
    return sorted({(u,v) for a,b in ((x,y),(y,x))
                   for u,v in ((a,b),(L-a,b),(a,L-b),(L-a,L-b))})

def load_points(path):
    data=json.loads(path.read_text());L,B=F(data['L']),F(data['B'])
    if B!=F(9977,10000) or L<=2*B or int(data['n'])<=0:
        raise ValueError('Require original-validator B, positive n, and L > 2B')
    weights=defaultdict(F)
    for x,y,w in data['atoms']:
        x,y,w=map(F,(x,y,w))
        if not (0<=x<=L and 0<=y<=L) or w<0: raise ValueError('Invalid atom')
        weights[x,y]+=w
    if not weights or sum(weights.values())<=0: raise ValueError('Empty or zero-mass input')
    groups={}
    for x,y in weights:
        o=orbit(x,y,L);key=o[0]
        if any(p not in weights or weights[p]!=weights[x,y] for p in o):
            raise ValueError('Input positions and weights must have exact D4 symmetry')
        groups[key]=o
    return data,[groups[k] for k in sorted(groups)]

class PointMaster:
    def __init__(self,L,groups,rhs,lp_timeout=120):
        self.L=F(L);self.B=F(9977,10000);self.groups=groups;self.rhs=rhs
        self.exact=[p for o in groups for p in o]
        self.pts=np.array([[float(x),float(y)] for x,y in self.exact])
        self.ids=np.repeat(np.arange(len(groups)),[len(o) for o in groups])
        self.sizes=np.array(list(map(len,groups)))
        self.dirs=load_validator(VENDOR).net()
        self.projections=[(self.pts[:,0]*float(c)+self.pts[:,1]*float(s),
                           -self.pts[:,0]*float(s)+self.pts[:,1]*float(c)) for t,c,s in self.dirs]
        self.keys=set();self.poses=[];self.A=sparse.csr_matrix((0,len(groups)))
        self.h=highspy.Highs()
        for key,val in [('output_flag',False),('threads',1),('solver','simplex'),('time_limit',lp_timeout),
                        ('primal_feasibility_tolerance',1e-9),('dual_feasibility_tolerance',1e-9)]:
            self.check(self.h.setOptionValue(key,val))
        n=len(groups)
        self.check(self.h.addCols(n,np.ones(n),np.zeros(n),np.full(n,np.inf),0,
                                 np.zeros(n+1,np.int32),np.array([],np.int32),np.array([],float)))
    @staticmethod
    def check(status):
        if status!=highspy.HighsStatus.kOk: raise RuntimeError(f'HiGHS: {status}')
    def row(self,r,x,y):
        r=int(r);x,y=F(str(x)),F(str(y));t,c,s=self.dirs[r]
        a=self.B*(c+s)/2
        # Tiny float oracle domain overshoots are clamped to the exact admissible boundary.
        if not (float(a)-1e-10<=float(x)<=float(self.L-a)+1e-10 and
                float(a)-1e-10<=float(y)<=float(self.L-a)+1e-10):
            raise ValueError('Centre outside domain')
        x,y=min(self.L-a,max(a,x)),min(self.L-a,max(a,y))
        au,av=self.projections[r]
        du=np.abs(au-float(x*c+y*s));dv=np.abs(av-float(-x*s+y*c));hb=float(self.B/2)
        included=(du<=hb)&(dv<=hb)
        # Resolve boundary ambiguity exactly; ordinary interior membership uses float.
        close=np.flatnonzero((np.abs(du-hb)<1e-10)|(np.abs(dv-hb)<1e-10))
        for i in close:
            px,py=self.exact[i]
            included[i]=(abs((px-x)*c+(py-y)*s)<=self.B/2 and
                         abs(-(px-x)*s+(py-y)*c)<=self.B/2)
        counts=np.bincount(self.ids[included],minlength=len(self.groups))
        return counts, [r,str(x),str(y)]
    def add_rows(self,poses):
        rows=[]
        for r,x,y in poses:
            counts,pose=self.row(r,x,y);key=counts.astype(np.uint8).tobytes()
            if key in self.keys: continue
            self.keys.add(key);self.poses.append(pose);rows.append(counts/self.sizes)
        if not rows: return 0
        block=sparse.csr_matrix(np.array(rows))
        self.check(self.h.addRows(len(rows),np.full(len(rows),self.rhs),np.full(len(rows),np.inf),
                                 block.nnz,block.indptr.astype(np.int32),block.indices.astype(np.int32),block.data))
        self.A=sparse.vstack([self.A,block],format='csr')
        return len(rows)
    def solve(self):
        run_status=self.h.run()
        if run_status==highspy.HighsStatus.kError: raise RuntimeError('HiGHS run error')
        status=self.h.getModelStatus()
        if status!=highspy.HighsModelStatus.kOptimal: return None,str(status)
        sol=self.h.getSolution();w=np.maximum(0,np.array(sol.col_value));d=np.array(sol.row_dual)
        minimum=float(np.min(self.A@w))
        # Simplex roundoff can leave coverage marginally short of rhs; rescale
        # weights uniformly to restore exact feasibility (mass only moves up).
        if 0<minimum<self.rhs:
            w=w*(self.rhs/minimum);minimum=float(np.min(self.A@w))
        gap=float(w.sum()-self.rhs*d.sum())
        # Dual-side roundoff has no cheap exact fix like the primal rescale above;
        # observed noise on this engine's runs tops out around 1.2e-5 (violation)
        # and 3.8e-5 (gap), so these keep margin over that without going slack.
        if minimum<self.rhs-2e-7 or abs(gap)>1e-4 or np.max(self.A.T@d)>1+2e-5:
            raise RuntimeError('LP residual failure')
        return w,dict(mass=float(w.sum()),min_row_mass=minimum,duality_gap=gap,rows=self.A.shape[0])
    def export(self,source,w):
        atoms=[]
        # Upward 1e-8 quantization, stabilized under the original float/ceil conversion.
        for group,mass in zip(self.groups,w):
            k=math.ceil(F.from_float(float(mass))/len(group)*10**8)
            while math.ceil(float(F(k,10**8))*10**8)>k:
                k+=1
            q=F(k,10**8)
            if q:
                atoms.extend([[str(x),str(y),str(q)] for x,y in group])
        total=sum((F(a[2]) for a in atoms),F(0))
        return dict(n=source['n'],L=str(self.L),B=str(self.B),net=source.get('net'),atoms=atoms,
                    total_mass=str(total),num_atoms=len(atoms),status='UNVERIFIED_REOPTIMIZED_POINTS')

def oracle_worker(candidate,out,max_fails,split,threshold,start_direction=0):
    data=json.loads(candidate.read_text());v=load_validator(VENDOR)
    pts=np.array([[float(F(x)),float(F(y))] for x,y,w in data['atoms']])
    # Same rationalise operation as original CLI; separation and final verifier see same weights.
    w=np.array([float(q) for q in v.rationalise([float(F(w)) for x,y,w in data['atoms']],
                                             [1]*len(pts),data['n'])[0]])
    with out.open('w') as stream:
        dirs=v.net()
        for offset in range(201):
            r=(start_direction+offset)%201;t,c,s=dirs[r]
            fails,boxes=v.bb_direction(pts,w,float(F(data['L'])),float(c),float(s),
                                      max_fails=max_fails,split=split,per_cell=1,thresh=threshold)
            stream.write(json.dumps(dict(direction=r,boxes=boxes,fails=fails))+'\n');stream.flush()

def separate(candidate,out,seconds,max_fails,split,threshold,start_direction=0):
    start=time.monotonic()
    with out.with_suffix('.log').open('w') as log:
        try:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--oracle-worker',str(candidate.resolve()),
                              str(out.resolve()),str(max_fails),str(split),str(threshold),str(start_direction)],
                             stdout=log,stderr=subprocess.STDOUT,timeout=seconds)
            status='COMPLETE' if p.returncode==0 else 'ERROR'
        except subprocess.TimeoutExpired: status='TIMEOUT'
    records=[]
    if out.exists():
        for line in out.read_text().splitlines():
            try: records.append(json.loads(line))
            except json.JSONDecodeError: break
    poses=[[d['direction'],x,y] for d in records for x,y,m in d['fails']]
    return poses,dict(status=status,start_direction=start_direction,directions=len(records),boxes=sum(d['boxes'] for d in records),
                      failing_boxes=len(poses),seconds=time.monotonic()-start)

def run(a):
    if a.rounds<1 or a.seed_grid<2 or a.max_fails<1 or a.split<1 or not 1+1e-7<a.rhs<1.1:
        raise ValueError('Invalid search limits or rhs')
    if not all(math.isfinite(t) and t>0 for t in (a.oracle_timeout,a.validator_timeout,a.lp_timeout)):
        raise ValueError('Timeouts must be finite and positive')
    if a.out.exists() and any(a.out.iterdir()): raise ValueError('Output directory must be new/empty')
    data,groups=load_points(a.candidate);a.out.mkdir(parents=True,exist_ok=True)
    master=PointMaster(data['L'],groups,a.rhs,a.lp_timeout)
    history=[];summary=dict(status='RUNNING',input_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
                           fixed_positions=True,engine_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           validator_sha256=hashlib.sha256(VENDOR.read_bytes()).hexdigest(),
                           settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},orbits=len(groups),input_atoms=len(data['atoms']),rhs=a.rhs,history=history)
    # Start from supplied point weights: use its deficits before the first optimization.
    initial=a.out/'initial.json';initial.write_text(json.dumps(data))
    poses=[]
    for r,(t,c,s) in enumerate(master.dirs):
        low=master.B*(c+s)/2
        # Cell interiors avoid initially imposing only event-boundary configurations.
        for u in (np.arange(a.seed_grid)+.5)/a.seed_grid:
            for v in (np.arange(a.seed_grid)+.5)/a.seed_grid:
                poses.append([r,str(low+F(str(u))*(master.L-2*low)),str(low+F(str(v))*(master.L-2*low))])
    master.add_rows(poses)
    if a.constraints:
        master.add_rows(json.loads(a.constraints.read_text()))
    if a.seed_poses:
        with np.load(a.seed_poses,allow_pickle=False) as archive:
            normalized=archive['poses']
        if normalized.ndim!=2 or normalized.shape[1]!=3 or not np.isfinite(normalized).all():
            raise ValueError('Invalid normalized seed poses')
        extra=[]
        for u,v,theta in normalized:
            if abs(u)>1 or abs(v)>1 or not 0<=theta<=1:
                raise ValueError('Invalid normalized seed pose')
            r=min(200,max(0,round(math.tan(theta*math.pi/8)/float(F(83,40000)))))
            t,c,s=master.dirs[r];extent=(master.L-master.B*(c+s))/2
            extra.append([r,str(master.L/2+F(str(u))*extent),str(master.L/2+F(str(v))*extent)])
        master.add_rows(extra)
    added,scan=separate(initial,a.out/'initial_oracle.jsonl',a.oracle_timeout,a.max_fails,a.split,1+1e-9)
    master.add_rows(added);summary['initial_scan']=scan
    candidate=None
    for iteration in range(a.rounds):
        w,stats=master.solve()
        if w is None:
            summary.update(status='LP_INFEASIBLE_OR_FAILED',solver_status=stats);break
        candidate=a.out/f'iteration_{iteration:03d}.json'
        exported=master.export(data,w);candidate.write_text(json.dumps(exported,separators=(',',':'))+'\n')
        stats.update(iteration=iteration,export_mass=float(F(exported['total_mass'])),active_atoms=len(exported['atoms']))
        print(json.dumps(stats),flush=True);history.append(stats)
        summary['latest_candidate']=candidate.name
        # A restricted numerical LP above budget means this fixed support needs reconsideration.
        if stats['mass']>=data['n']:
            summary['status']='LP_BUDGET_EXCEEDED';break
        poses,scan=separate(candidate,a.out/f'oracle_{iteration:03d}.jsonl',a.oracle_timeout,a.max_fails,a.split,
                            1+(a.rhs-1)/2,start_direction=(67+iteration*53)%201)
        stats['scan']=scan;new=master.add_rows(poses);stats['new_rows']=new
        summary['status']='ROUND_LIMIT'
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if scan['status']=='ERROR': summary['status']='ORACLE_ERROR';break
        if new==0:
            summary['status']='SEPARATION_CLEAR' if scan['status']=='COMPLETE' and not poses else 'SEPARATION_STALLED'
            break
    # Check the actual exported file with the unmodified original CLI, regardless of search result.
    if candidate is not None:
        result=run_validator(candidate,VENDOR,a.validator_timeout);summary['validation']=result
        if result['status']=='VERIFIED_BY_ORIGINAL_FLOAT_VALIDATOR': summary['status']=result['status']
    (a.out/'constraints.json').write_text(json.dumps(master.poses)+'\n')
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['status']=='VERIFIED_BY_ORIGINAL_FLOAT_VALIDATOR' else 2

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--oracle-worker':
        oracle_worker(Path(sys.argv[2]),Path(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),float(sys.argv[6]),int(sys.argv[7]));sys.exit(0)
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('candidate',type=Path)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--rounds',type=int,default=12)
    ap.add_argument('--constraints',type=Path,help='Previously saved constraints.json; keeps accumulated rows')
    ap.add_argument('--seed-poses',type=Path,help='Rectangle engine normalized constraints.npz')
    ap.add_argument('--rhs',type=float,default=1.0001);ap.add_argument('--seed-grid',type=int,default=5)
    ap.add_argument('--max-fails',type=int,default=16);ap.add_argument('--split',type=int,default=4)
    ap.add_argument('--lp-timeout',type=float,default=120)
    ap.add_argument('--oracle-timeout',type=float,default=60);ap.add_argument('--validator-timeout',type=float,default=120)
    sys.exit(run(ap.parse_args()))
