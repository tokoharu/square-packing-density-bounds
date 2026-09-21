"""a: propose larger L, restore coverage, run b/c, and promote only certificates."""
import argparse
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from certify import certify, density_digest
from geometry import Geometry, rectangle_key
from plot_solution import plot_solution


def atomic_json(path, data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def load_incumbent(path, N):
    """Check data/proof linkage; this detects stale artifacts, not malicious proofs."""
    path=Path(path);data=json.loads(path.read_text());meta=data.get('certificate',{})
    if not data.get('globally_verified') or meta.get('density_sha256')!=density_digest(data):
        raise ValueError('Incumbent must be an unchanged certified density')
    summary=json.loads((path.parent/'verification_summary.json').read_text())
    digest=hashlib.sha256((path.parent/'certificate_input.txt').read_bytes()).hexdigest()
    if (summary.get('status')!='VERIFIED' or summary.get('angle_cases')!=201 or
            summary.get('input_sha256')!=digest or meta.get('input_sha256')!=digest or
            summary.get('verifier_source_sha256')!=hashlib.sha256((path.parent/'verify.cpp').read_bytes()).hexdigest()):
        raise ValueError('Certificate artifacts do not match')
    exact=json.loads(path.read_text(),parse_float=F)
    mass=sum(map(F,exact['weights']))
    if mass!=F(meta['mass_exact']) or not mass < F(N) or F(exact['L'])!=F(meta['L']):
        raise ValueError('Incumbent mass/L mismatch or mass limit not met')
    return data


def transfer(parent, new_L, rhs=1.001):
    old=float(parent['L']);new_L=float(new_L)
    if not np.isfinite(new_L) or new_L<=old:
        raise ValueError('L must strictly increase')
    rectangles=[];weights=[];keys=set()
    def add(r,w):
        key=rectangle_key(r,new_L)
        if key in keys:return
        Geometry(new_L,parent['B'],[r])
        rectangles.append(np.asarray(r).tolist());weights.append(float(w));keys.add(key)
    active=[(np.asarray(r,float),w) for r,w in zip(parent['rectangles'],parent['weights']) if w>0]
    # Two anchors: relative position in K, and distance from the nearby walls.
    for r,w in active:add(r*(new_L/old),w)
    for r,w in active:add(r,0.)
    # A full interior rectangle prevents uncovered-support infeasibility.
    margin=F(1,1000);LF=F(str(new_L));BF=F(str(parent['B']))
    fallback=[float(margin),float(margin),float(LF-margin),float(LF-margin)]
    a,b,c,d=map(lambda x:F(str(x)),fallback)
    delta=max(a,b,LF-c,LF-d)
    # Any rotated B-square loses at most 4*sqrt(2)*B*delta <= 6*B*delta
    # to the container's four boundary strips.
    lower_area=BF*BF-6*BF*delta
    if lower_area<=0:raise ValueError('Fallback area bound is not positive')
    area=(c-a)*(d-b)
    needed=F(str(rhs))*area/lower_area
    unit=10**12
    mass=F((needed.numerator*unit+needed.denominator-1)//needed.denominator,unit)
    add(fallback,float(mass))
    if F(str(weights[-1]))*lower_area/area < F(str(rhs)):
        raise ValueError('Fallback decimal rounding lost the coverage bound')
    return {'L':new_L,'B':parent['B'],'rectangles':rectangles,'weights':weights,'rhs':rhs,
            'globally_verified':False,'status':'COVERAGE_RESTORED_MASS_BUDGET_NOT_ENFORCED',
            'transfer':{'parent_L':old,'methods':['scaled','wall_anchored'],
                        'fallback_index':len(rectangles)-1,'fallback_mass':str(mass),
                        'fallback_coverage_lower_bound':str(F(str(weights[-1]))*lower_area/area),
                        'initial_mass':sum(weights)}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--incumbent',type=Path,default=Path(__file__).parent/'runs/global_v1/certificate/certified_candidate.json')
    ap.add_argument('--poses',type=Path,default=Path(__file__).parent/'runs/global_v1/constraints.npz')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--N',type=int,default=26)
    ap.add_argument('--step',default='0.001')
    ap.add_argument('--attempts',type=int,default=1)
    ap.add_argument('--cycles',type=int,default=2)
    ap.add_argument('--proposal',choices=['free','long-split'],default='free')
    ap.add_argument('--global-rounds',type=int,default=40)
    args=ap.parse_args()
    step=F(args.step)
    if step<=0 or args.attempts<1 or args.N<1 or args.cycles<0:
        ap.error('Positive step/attempts/N and nonnegative cycles required')
    out=args.out.resolve();state_path=out/'advance_state.json'
    if args.resume:
        state=json.loads(state_path.read_text())
        if state['N']!=args.N:ap.error('N differs from saved run')
        for record in state['attempts']:
            if record['status']=='RUNNING':record['status']='INTERRUPTED_NOT_ACCEPTED'
        step=F(state['next_step'])
        atomic_json(state_path,state)
    else:
        if out.exists() and any(out.iterdir()):ap.error('Use a new/empty output, or --resume')
        out.mkdir(parents=True,exist_ok=True)
        initial=load_incumbent(args.incumbent,args.N)
        # Snapshot the certified parent so this run is independently portable.
        shutil.copytree(args.incumbent.resolve().parent,out/'initial_certificate',
                        ignore=shutil.ignore_patterns('verify','__pycache__'))
        shutil.copyfile(args.poses,out/'initial_constraints.npz')
        state={'N':args.N,'best_L':initial['L'],'best_candidate':'initial_certificate/'+args.incumbent.name,
               'best_poses':'initial_constraints.npz','next_step':str(step),'attempts':[]}
        atomic_json(state_path,state)
    for _ in range(args.attempts):
        parent_path=out/state['best_candidate'];parent=load_incumbent(parent_path,args.N)
        target=F(str(parent['L']))+step
        if float(target)<=parent['L']:
            print(json.dumps({'operation':'advance_stop','status':'STEP_BELOW_FLOAT_RESOLUTION','best_L':parent['L']}),flush=True)
            break
        attempt=out/f"attempt_{len(state['attempts']):03d}"
        attempt.mkdir()
        record={'L':float(target),'step':str(step),'parent_L':parent['L'],'directory':attempt.name,'status':'RUNNING'}
        state['attempts'].append(record);atomic_json(state_path,state)
        seed=transfer(parent,float(target));atomic_json(attempt/'initial_candidate.json',seed)
        shutil.copyfile(out/state['best_poses'],attempt/'initial_constraints.npz')
        # Normalized square poses are reinterpreted at new L. Old LP bases are NOT reused.
        cmd=[sys.executable,str(Path(__file__).with_name('engine.py')),
             '--seed',str(attempt/'initial_candidate.json'),'--poses',str(attempt/'initial_constraints.npz'),
             '--out',str(attempt/'search'),'--cycles',str(args.cycles),'--b-rounds','3',
             '--global-check','--global-rounds',str(args.global_rounds),'--proposal',args.proposal]
        print(json.dumps({'operation':'advance_start',**record}),flush=True)
        try:
            with (attempt/'search.log').open('w') as stream:
                subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True,
                               env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
            source=attempt/'search/candidate.json'
            exact=json.loads(source.read_text(),parse_float=F);mass=sum(map(F,exact['weights']))
            record['mass']=float(mass)
            if mass>=args.N:
                record['status']='MASS_LIMIT_NOT_MET'
            else:
                certify(source,attempt/'certificate')
                accepted=attempt/'certificate/certified_candidate.json'
                checked=load_incumbent(accepted,args.N)
                record['status']='ACCEPTED'
                state.update(best_L=checked['L'],best_candidate=str(accepted.relative_to(out)),
                             best_poses=str((attempt/'search/constraints.npz').relative_to(out)))
                # Commit the new best before optional plotting, so plotting cannot lose it.
                atomic_json(state_path,state)
                plot_solution(accepted,attempt/'solution.png')
        except subprocess.CalledProcessError as error:
            record['status']='SEARCH_OR_PROOF_INCOMPLETE';record['returncode']=error.returncode
        if record['status']!='ACCEPTED':step/=2
        state['next_step']=str(step);atomic_json(state_path,state)
        print(json.dumps({'operation':'advance_result',**record,'best_L':state['best_L']}),flush=True)


if __name__=='__main__':main()
