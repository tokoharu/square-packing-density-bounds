"""Recheck an original-validator unresolved centre using exact rational arithmetic."""
import argparse
from fractions import Fraction as F
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

def diagnose(path, validator, direction=0):
    spec=importlib.util.spec_from_file_location('original_point_validator',validator)
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    data=json.loads(path.read_text());L=F(data['L']);B=F(data['B'])
    if B != v.B: raise ValueError('B mismatch')
    atoms=[tuple(map(F,a)) for a in data['atoms']]
    rounded=[F(math.ceil(float(w)*v.WDEN),v.WDEN) for x,y,w in atoms]
    t,c,s=v.net()[direction]
    fails,nb=v.bb_direction(np.array([[float(x),float(y)] for x,y,w in atoms]),
                           np.array(list(map(float,rounded))),float(L),float(c),float(s),
                           max_fails=1,thresh=1+1e-9)
    result=dict(direction=direction,boxes=nb,status='NO_WITNESS_FROM_FIRST_BOX')
    if fails:
        x,y,lower=fails[0];x,y=F.from_float(x),F.from_float(y)
        a=B*(c+s)/2
        if not (a<=x<=L-a and a<=y<=L-a):
            result['status']='FLOAT_CENTRE_OUTSIDE_EXACT_DOMAIN'
            return result
        selected=[i for i,(px,py,w) in enumerate(atoms)
                  if abs((px-x)*c+(py-y)*s)<=B/2 and abs(-(px-x)*s+(py-y)*c)<=B/2]
        mass=sum((atoms[i][2] for i in selected),F(0))
        rmass=sum((rounded[i] for i in selected),F(0))
        result.update(status='EXACT_B_SQUARE_DEFICIT' if rmass<1 else 'UNRESOLVED_BOX_ONLY',
                      center=[str(x),str(y)],cos=str(c),sin=str(s),B=str(B),
                      mass_exact=str(mass),mass_decimal=float(mass),
                      validator_rounded_mass_exact=str(rmass),validator_rounded_mass_decimal=float(rmass),
                      note='Deficit in a contained B-square; this alone does not disprove unit-square coverage.')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('candidate',type=Path)
    ap.add_argument('--validator',type=Path,default=Path(__file__).parent/'vendor/point_validator/verify.py')
    ap.add_argument('--direction',type=int,choices=range(201),default=0)
    a=ap.parse_args();result=diagnose(a.candidate,a.validator,a.direction)
    a.candidate.with_suffix('.witness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
