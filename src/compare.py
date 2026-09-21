"""Compare old/new dictionaries on exactly the same final rows; numerical only."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import scipy
import numba
import highspy
from master import Master


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--seed', type=Path, default=Path(__file__).parent/'examples/seed_20.json')
    args = ap.parse_args()
    old = json.loads(args.seed.read_text())
    final = json.loads((args.run/'candidate.json').read_text())
    if old['L'] != final['L'] or old['B'] != final['B']:
        raise ValueError('Comparison requires identical L and B')
    poses = np.load(args.run/'constraints.npz')['poses']
    results = []
    for name, rects in [('seed_dictionary', old['rectangles']), ('new_dictionary', final['rectangles'])]:
        master = Master(final['L'], final['B'], rects, final['rhs'])
        master.add_rows(poses)
        sol = master.solve()
        results.append({'dictionary': name, 'columns': len(rects), 'rows': len(poses),
                        'active': int(np.count_nonzero(sol.weights > 0)), 'mass': sol.mass,
                        'lp_seconds_cold': sol.seconds, 'iterations_cold':sol.iterations,
                        'duality_gap': sol.duality_gap})
    result = {'status':'FINITE_CONSTRAINT_COMPARISON_ONLY', 'L': final['L'],
              'B':final['B'], 'rhs':final['rhs'], 'results':results,
              'mass_improvement_same_final_rows':results[0]['mass']-results[1]['mass'],
              'candidate_sha256':hashlib.sha256((args.run/'candidate.json').read_bytes()).hexdigest(),
              'environment':{'python':platform.python_version(), 'numpy':np.__version__,
                             'scipy':scipy.__version__, 'numba':numba.__version__,
                             'highspy':highspy.Highs().version()}}
    (args.run/'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
