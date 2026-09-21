"""Integrate a D4 rectangle density into rational weighted grid points."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


WDEN = 10 ** 8  # matches the original validator's weight-rounding denominator


def rescale_factor(original, num_atoms, n):
    """Largest uniform boost to every atom weight that still keeps the WDEN-rounded
    total strictly below n, after accounting for the up-to-1/WDEN rounding-up each
    atom can separately gain in the original validator. Returns 1 (no boost) when
    there is no safe headroom left to use (e.g. original is already close to n)."""
    headroom = F(n) - F(num_atoms + 1, WDEN)
    if headroom <= original:
        return F(1)
    return headroom / original


def cdf(x, a, b, eps):
    """Exact CDF of Uniform[a,b] + Uniform[-eps,eps]."""
    if eps == 0:
        return min(F(1), max(F(0), (x-a)/(b-a)))
    if x <= a-eps: return F(0)
    if x >= b+eps: return F(1)
    positive_square = lambda t: max(F(0), t)**2
    return (positive_square(x-a+eps)-positive_square(x-b+eps)
            -positive_square(x-a-eps)+positive_square(x-b-eps))/(4*eps*(b-a))


def axis_cells(a, b, eps, step, count):
    lo = max(0, (a-eps)//step)
    hi = min(count, math.ceil((b+eps)/step))
    out = []
    for i in range(lo, hi):
        mass = cdf((i+1)*step,a,b,eps)-cdf(i*step,a,b,eps)
        if mass: out.append((i,mass))
    return out


def convert(data, spacing, epsilon=F(1,20000), n=26, max_cells=4_000_000, rescale=True, rescale_to=None):
    L, B, spacing, epsilon = map(F, (data['L'], data['B'], spacing, epsilon))
    if L <= 0 or not 0 < B < 1 or spacing <= 0 or epsilon < 0 or n <= 0:
        raise ValueError('Require L, spacing, n > 0, 0 < B < 1 and epsilon >= 0')
    if len(data['rectangles']) != len(data['weights']):
        raise ValueError('Mismatched rectangles and weights')
    # Even cell count preserves all D4 reflections exactly; step is at most requested spacing.
    count = 2*math.ceil(L/(2*spacing))
    if count*count > max_cells:
        raise ValueError(f'Grid has {count*count} cells, exceeds --max-cells {max_cells}')
    step = L/count
    weights = defaultdict(F)
    original = F(0)
    for row, mass in zip(data['rectangles'], data['weights']):
        mass = F(mass)
        if mass < 0: raise ValueError('Negative weight')
        if not mass: continue
        a,b,c,d = map(F,row)
        if not (epsilon <= a < c <= L-epsilon and epsilon <= b < d <= L-epsilon):
            raise ValueError('Smoothed rectangle must lie inside container')
        original += mass
        xs,ys = axis_cells(a,c,epsilon,step,count),axis_cells(b,d,epsilon,step,count)
        for i,wx in xs:
            for j,wy in ys:
                w = mass*wx*wy/8
                for u,v in ((i,j),(j,i)):
                    for ii,jj in ((u,v),(count-1-u,v),(u,count-1-v),(count-1-u,count-1-v)):
                        weights[ii,jj] += w
    if not original: raise ValueError('Zero mass')
    assert sum(weights.values()) == original
    num_atoms = sum(1 for w in weights.values() if w)
    if rescale_to is not None:
        target = F(rescale_to)
        if target <= 0: raise ValueError('--rescale-to must be positive')
        if target <= original:
            raise ValueError(f'--rescale-to={target} must exceed the current total mass {original}; '
                              'a smaller target would shrink weights, which only increases deficits')
        factor = target/original
    elif rescale:
        factor = rescale_factor(original, num_atoms, n)
    else:
        factor = F(1)
    if factor != 1:
        # Uniform D4-symmetric boost: every cell (and so every B-square's covered mass,
        # which is just a sum of a subset of these weights) scales by the same factor,
        # buying back margin the grid-midpoint discretization may have cost near boundaries.
        for key in list(weights):
            weights[key] *= factor
    scaled_total = original*factor
    atoms = [[str((F(i)+F(1,2))*step),str((F(j)+F(1,2))*step),str(w)]
             for (i,j),w in sorted(weights.items()) if w]
    # This mirrors the original validator's float -> ceil conversion, including float effects.
    checked_mass = sum((F(math.ceil(float(F(a[2]))*WDEN),WDEN) for a in atoms),F(0))
    cert = dict(n=n,L=str(L),B=str(B),net='theta_r = 2*atan(r*83/40000), r=0..200',
                atoms=atoms,num_atoms=len(atoms),total_mass=str(scaled_total),
                status='UNVERIFIED_POINT_CANDIDATE')
    meta = dict(requested_spacing=str(spacing),actual_spacing=str(step),grid_cells_per_axis=count,
                epsilon=str(epsilon),method='exact cell mass at cell midpoint; even uniform D4 grid',
                mass_exact=str(scaled_total),mass_decimal=float(scaled_total),
                validator_rounded_mass_exact=str(checked_mass),validator_rounded_mass_decimal=float(checked_mass),
                validator_rounding_increase=str(checked_mass-scaled_total),num_atoms=len(atoms),
                rescaled=factor!=1,rescale_factor=str(factor),
                original_mass_exact=str(original),original_mass_decimal=float(original))
    return cert,meta


def run_validator(candidate, validator, timeout):
    data=json.loads(candidate.read_text())
    if F(data['B']) != F(9977,10000):
        raise ValueError('Original validator hardcodes B=9977/10000; incompatible candidate')
    log=candidate.with_suffix('.validator.log')
    start=time.monotonic()
    with log.open('w') as stream:
        try:
            result=subprocess.run([sys.executable,'-u',str(validator.resolve()),str(candidate.resolve())],
                                  stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,check=False)
            returncode=result.returncode
            lines=log.read_text().splitlines()
            status=('VERIFIED_BY_ORIGINAL_FLOAT_VALIDATOR' if returncode==0 and lines and lines[-1]=='VERIFIED'
                    else 'NOT_VERIFIED' if returncode==0 and lines and lines[-1]=='NOT VERIFIED' else 'ERROR')
        except subprocess.TimeoutExpired:
            status,returncode='TIMEOUT',None
    report=dict(status=status,returncode=returncode,seconds=time.monotonic()-start,
                candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(validator.read_bytes()).hexdigest(),
                log_file=log.name,validator=str(validator),
                note='Original float64 checker; NOT_VERIFIED can include unresolved boxes, not just counterexamples.')
    candidate.with_suffix('.validator.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('candidate',type=Path)
    ap.add_argument('--spacing',type=F,required=True,help='Maximum grid spacing; e.g. 1/64 or 0.01')
    ap.add_argument('--epsilon',type=F,default=F(1,20000),help='Smoothing width; 0 selects raw rectangle density')
    ap.add_argument('--n',type=int,default=26)
    ap.add_argument('--max-cells',type=int,default=4_000_000)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--validate',action='store_true')
    ap.add_argument('--validator',type=Path,default=Path(__file__).parent/'vendor/point_validator/verify.py')
    ap.add_argument('--timeout',type=float,default=120,help='Validator time limit in seconds')
    ap.add_argument('--rescale-to',type=F,default=None,
                     help='Uniformly rescale every weight so the exact total becomes this many units '
                          '(must exceed the current total). Overrides the automatic near-N rescaling.')
    ap.add_argument('--no-rescale',action='store_true',
                     help='Keep weights exactly as computed from the source density: skip the automatic '
                          'uniform rescaling toward N that is otherwise applied by default.')
    a=ap.parse_args()
    if a.timeout <= 0 or not math.isfinite(a.timeout): ap.error('--timeout must be finite and positive')
    if a.out.suffix != '.json': ap.error('--out must end in .json')
    if a.rescale_to is not None and a.no_rescale: ap.error('--rescale-to and --no-rescale are mutually exclusive')
    if any(p.exists() for p in [a.out,a.out.with_suffix('.metadata.json'),a.out.with_suffix('.validator.json'),a.out.with_suffix('.validator.log')]):
        ap.error('Output or companion file already exists; choose a new --out')
    data=json.loads(a.candidate.read_text(),parse_float=F)
    cert,meta=convert(data,a.spacing,a.epsilon,a.n,a.max_cells,rescale=not a.no_rescale,rescale_to=a.rescale_to)
    meta['source_sha256']=hashlib.sha256(a.candidate.read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(cert,separators=(',',':'))+'\n')
    meta['point_sha256']=hashlib.sha256(a.out.read_bytes()).hexdigest()
    a.out.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)
    if a.validate:
        report=run_validator(a.out,a.validator,a.timeout)
        print(json.dumps(report,indent=2))
        if report['status'] != 'VERIFIED_BY_ORIGINAL_FLOAT_VALIDATOR': return 2
    return 0

if __name__=='__main__': sys.exit(main())
