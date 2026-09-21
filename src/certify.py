"""Exact-decimal certificate input, preserving candidate weights (no rescaling)."""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


def density_digest(data):
    payload={key:data[key] for key in ['L','B','rectangles','weights']}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def enclosing(x):
    v = float(x); exact = F.from_float(v)
    lo = v if exact <= x else math.nextafter(v,-math.inf)
    hi = v if exact >= x else math.nextafter(v,math.inf)
    if not F.from_float(lo) <= x <= F.from_float(hi):
        raise ValueError('Interval conversion failed')
    return lo.hex()+' '+hi.hex()


def prepare(source, out):
    data = json.loads(source.read_text(),parse_float=F)
    L,B = F(data['L']),F(data['B'])
    epsilon,D = F(1,20000),F(83,40000)
    if not (L*L > 2*B*B and 0 < B < 1 and B*(1+D)+3*epsilon < 1):
        raise ValueError('Invalid container or angular-net/smoothing safety condition')
    if len(data['rectangles']) != len(data['weights']):
        raise ValueError('Mismatched rectangles and weights')
    rectangles = []; total=F(0)
    for row, mass in zip(data['rectangles'],data['weights']):
        mass = F(mass)
        if mass < 0: raise ValueError('Negative weight')
        if not mass: continue
        x0,y0,x1,y1 = map(F,row)
        if not epsilon < x0 < x1 < L-epsilon or not epsilon < y0 < y1 < L-epsilon:
            raise ValueError('Support must stay inside the container after smoothing')
        total += mass
        rho = mass/8/((x1-x0)*(y1-y0))
        for swap in [False,True]:
            a,b,c,e = (y0,x0,y1,x1) if swap else (x0,y0,x1,y1)
            for sx,sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                u,v = (a,c) if sx==1 else (L-c,L-a)
                z,t = (b,e) if sy==1 else (L-e,L-b)
                rectangles.append((u,z,v,t,rho))
    if not total: raise ValueError('Zero total mass')
    centers={L/2,L-B/2}
    for r in rectangles:
        for end in [r[0],r[2]]:
            for sign in [-1,1]:
                center=end+sign*B/2
                if L/2 <= center <= L-B/2: centers.add(center)
    lines=[enclosing(L),enclosing(B),str(len(rectangles))]
    lines.extend(' '.join(enclosing(x) for x in r) for r in rectangles)
    lines.append(str(len(centers))); lines.extend(enclosing(x) for x in sorted(centers))
    out.mkdir(parents=True,exist_ok=True)
    (out/'certificate_input.txt').write_text('\n'.join(lines)+'\n')
    meta={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'density_sha256':density_digest(json.loads(source.read_text())),
          'input_sha256':hashlib.sha256((out/'certificate_input.txt').read_bytes()).hexdigest(),
          'L':str(L),'B':str(B),'mass_exact':str(total),'mass_decimal':float(total),
          'epsilon':str(epsilon),'D':str(D),'angle_count':201,
          'safety_exact':str(B*(1+D)+3*epsilon),'rescaled':False}
    (out/'certificate_metadata.json').write_text(json.dumps(meta,indent=2))
    return meta


def certify(source, out, workers=4):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Certificate directory must be new/empty')
    meta=prepare(source,out)
    for name in ['verify.cpp','run_verify.py']:
        shutil.copyfile(Path(__file__).with_name(name),out/name)
    subprocess.run([sys.executable,'run_verify.py','--workers',str(workers)],cwd=out,check=True)
    report=json.loads((out/'verification_summary.json').read_text())
    if (report['status']!='VERIFIED' or report['angle_cases']!=201 or
            report['input_sha256']!=meta['input_sha256']):
        raise RuntimeError('Certificate summary mismatch')
    # Only this separate artifact receives the verified status.
    data=json.loads(source.read_text())
    data.update(status='VERIFIED_CONTINUOUS_DENSITY',globally_verified=True,
                certificate=meta,coverage_lower_bound_exact='10001/10000')
    (out/'certified_candidate.json').write_text(json.dumps(data,indent=2))
    return meta


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('candidate',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();print(json.dumps(certify(a.candidate.resolve(),a.out.resolve(),a.workers)))
