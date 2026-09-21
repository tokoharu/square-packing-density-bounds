"""Controlled long-axis refinement experiment at fixed L, then full certification."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from master import Master
from refinement import propose_splits
from pricing import propose
from advance import load_incumbent


def main():
    root=Path(__file__).parent
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=Path,default=root/'runs/advance_v2/attempt_000/certificate/certified_candidate.json')
    ap.add_argument('--poses',type=Path,default=root/'runs/advance_v2/attempt_000/search/constraints.npz')
    ap.add_argument('--parents',type=int,default=4)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--certify',action='store_true')
    args=ap.parse_args();out=args.out.resolve()
    if out.exists() and any(out.iterdir()):ap.error('Use a new/empty output')
    out.mkdir(parents=True,exist_ok=True)
    data=load_incumbent(args.seed,26);poses=np.load(args.poses)['poses']
    def make():
        m=Master(data['L'],data['B'],data['rectangles'],data.get('rhs',1.001))
        m.add_rows(poses);return m
    master=make();baseline=master.solve()
    selected,ranking=propose_splits(master,baseline,args.parents)
    (out/'split_ranking.json').write_text(json.dumps({'selected':selected,'all':ranking},indent=2))
    if not selected:raise RuntimeError('No improving split proposed')
    added=master.add_columns([r for item in selected for r in item['children']])
    improved=master.solve()
    seed={'L':master.L,'B':master.B,'rhs':master.rhs,'rectangles':master.rectangles.tolist(),
          'weights':improved.weights.tolist(),'globally_verified':False,'status':'FINITE_LP_ONLY'}
    (out/'split_seed.json').write_text(json.dumps(seed,indent=2))
    np.savez_compressed(out/'initial_constraints.npz',poses=master.poses)
    # Same row set and same number of added columns for the free-position control.
    control=make();control_sol=control.solve()
    candidates,pricing_report=propose(control,control_sol,max_columns=added,seed=8261)
    control_added=control.add_columns([c['rectangle'] for c in candidates]) if candidates else 0
    control_final=control.solve()
    report={'L':master.L,'baseline_mass':baseline.mass,'baseline_certified_mass':sum(data['weights']),
            'fixed_rows':len(poses),'split_parents':len(selected),'split_added_columns':added,
            'split_mass_same_rows':improved.mass,'split_gain_same_rows':baseline.mass-improved.mass,
            'free_added_columns':control_added,'free_mass_same_rows':control_final.mass,
            'free_gain_same_rows':baseline.mass-control_final.mass,'free_pricing':pricing_report,
            'free_comparison_globally_verified':False}
    (out/'study.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if args.certify:
        subprocess.run([sys.executable,str(root/'engine.py'),'--seed',str(out/'split_seed.json'),
                        '--poses',str(out/'initial_constraints.npz'),'--out',str(out/'repaired'),
                        '--cycles','0','--b-rounds','3','--global-rounds','40',
                        '--global-check','--certify','--plot'],check=True)
        final=json.loads((out/'repaired/certificate/certified_candidate.json').read_text())
        w=np.asarray(final['weights']);report.update(certified_mass=float(w.sum()),
            certified_gain=sum(data['weights'])-float(w.sum()),active_rectangles=int(np.count_nonzero(w>0)),
            coverage_lower_bound_exact=final['coverage_lower_bound_exact'],globally_verified=True)
        details=[];offset=len(data['rectangles'])
        # Both children are new in the actual trial; explicit coordinate lookup also
        # handles duplicate columns without silently assigning the wrong weights.
        for item in selected:
            children=[]
            for r in item['children']:
                indices=[i for i,q in enumerate(final['rectangles']) if np.allclose(q,r,rtol=0,atol=1e-12)]
                children.append(sum(w[i] for i in indices))
            p=item['parent_index'];parent_mass=float(w[p])
            effective=np.asarray(children)+parent_mass*np.asarray(item['fractions'])
            details.append({**item,'retained_parent_mass':parent_mass,'child_masses':children,
                            'effective_child_masses':effective.tolist()})
        report['final_split_details']=details
        (out/'study.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
