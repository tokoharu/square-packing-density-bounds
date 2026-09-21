"""Fixed-L b/c research engine. Does not update a certified incumbent or perform a."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from master import Master
from pricing import propose
from separation import separate


def summary(solution):
    d = asdict(solution)
    del d['weights']; del d['dual']
    d['active_rectangles'] = int(np.count_nonzero(solution.weights > 0))
    return d


def checkpoint(out, master, solution, history, config):
    d = {'L': master.L, 'B': master.B, 'rhs': master.rhs,
         'rectangles': master.rectangles.tolist(), 'weights': solution.weights.tolist(),
         'mass': solution.mass, 'status': 'FINITE_LP_OPTIMAL_NOT_GLOBALLY_VERIFIED',
         'globally_verified': False,
         'last_validation': next((r for r in reversed(history) if r['operation']=='independent_validation'), None),
         'history': history, 'config': config}
    # Atomic JSON replacement; separate runs must use separate output directories.
    temp = out/'candidate.tmp'
    temp.write_text(json.dumps(d, indent=2)); temp.replace(out/'candidate.json')
    np.savez_compressed(out/'constraints.npz', poses=master.poses)
    np.savez_compressed(out/'dual.npz', poses=master.poses, dual=solution.dual)
    if master.h is not None:
        master.save_basis(out/'basis.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=Path, default=Path(__file__).parent/'examples/seed_20.json')
    ap.add_argument('--poses', type=Path, default=Path(__file__).parent/'examples/initial_poses.npz')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--backend', choices=['highs', 'scipy'], default='highs')
    ap.add_argument('--basis', type=Path, help='Optional saved basis, with matching candidate/constraints')
    ap.add_argument('--cycles', type=int, default=3)
    ap.add_argument('--b-rounds', type=int, default=3)
    ap.add_argument('--rhs', type=float, default=1.001)
    ap.add_argument('--pricing-power', type=int, default=9)
    ap.add_argument('--separation-power', type=int, default=12)
    ap.add_argument('--columns', type=int, default=12)
    ap.add_argument('--proposal',choices=['free','long-split'],default='free')
    ap.add_argument('--split-parents',type=int,default=4)
    ap.add_argument('--split-axis',choices=['long','short'],default='long',
                     help='long-split only: bisect each candidate rectangle\'s longer (default) or shorter side')
    ap.add_argument('--short-floor',type=float,default=1e-3,
                     help='--split-axis=short only: below this shorter-side width, fall back to the long side')
    ap.add_argument('--validation-rounds', type=int, default=3)
    ap.add_argument('--widths', type=float, nargs='+', default=[1/64,1/32,1/16,1/8,1/4,1/2])
    ap.add_argument('--random-seed', type=int, default=2601)
    ap.add_argument('--global-check', action='store_true', help='Repair via full-center angle-net screening')
    ap.add_argument('--global-rounds', type=int, default=30)
    ap.add_argument('--screen-budget', type=int, default=1000000)
    ap.add_argument('--certify', action='store_true', help='Run outward-rounded proof after global repair')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    if args.certify and not args.global_check:
        ap.error('--certify requires --global-check')
    if args.global_rounds < 0 or args.screen_budget < 1:
        ap.error('Invalid global screening budget')
    if args.cycles < 0 or args.b_rounds < 1 or args.columns < 1 or args.validation_rounds < 0:
        ap.error('cycles/validation-rounds >= 0, b-rounds/columns >= 1 required')
    if args.out.exists() and any(args.out.iterdir()):
        ap.error('Use a new/empty output directory to preserve earlier checkpoints')
    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['seed_sha256'] = hashlib.sha256(args.seed.read_bytes()).hexdigest()
    data = json.loads(args.seed.read_text())
    master = Master(data['L'], data['B'], data['rectangles'], args.rhs, args.backend)
    master.add_rows(np.load(args.poses)['poses'])
    if args.basis:
        if master.h is None:
            ap.error('--basis requires highs backend')
        master.load_basis(args.basis)
    history = []
    started = perf_counter()
    def emit(record):
        record['elapsed_seconds'] = perf_counter()-started
        record['rows'] = len(master.poses)
        record['rectangles'] = len(master.rectangles)
        history.append(record)
        print(json.dumps(record), flush=True)
    sol = master.solve()
    emit({'operation':'initial_b', **summary(sol)})
    checkpoint(args.out, master, sol, history, config)

    # A finite b budget allows c to improve support before exact separation converges.
    for cycle in range(args.cycles+1):
        for j in range(args.b_rounds):
            more, report = separate(master, sol, args.random_seed+100*cycle+j,
                                    args.separation_power)
            emit({'operation':'separation', 'cycle':cycle, **report})
            if not len(more):
                break
            before = sol.mass
            master.add_rows(more)
            sol = master.solve()
            if sol.mass < before-1e-6:
                raise RuntimeError('Adding rows unexpectedly decreased optimal mass')
            emit({'operation':'b_add_rows', 'cycle':cycle, **summary(sol)})
            checkpoint(args.out, master, sol, history, config)
        checkpoint(args.out, master, sol, history, config)
        if cycle == args.cycles:
            break
        if args.proposal=='long-split':
            from refinement import propose_splits
            selected,ranking=propose_splits(master,sol,args.split_parents,
                                             split_axis=args.split_axis,short_floor=args.short_floor)
            candidates=[{'rectangle':r,'score':score,'parent_index':item['parent_index']}
                        for item in selected for r,score in zip(item['children'],item['child_scores'])]
            report={'status':'LONG_AXIS_SPLIT_PROPOSALS','selected_parents':selected,
                    'eligible_parents':len(ranking)}
        else:
            candidates, report = propose(master, sol, args.random_seed+cycle,
                                         args.pricing_power, max_columns=args.columns, widths=args.widths)
        emit({'operation':'c_propose', 'cycle':cycle, **report, 'proposals':candidates})
        if not candidates:
            # Heuristic failure is not optimality; later cycles try new random seeds.
            checkpoint(args.out, master, sol, history, config)
            continue
        before = sol.mass
        count = master.add_columns([p['rectangle'] for p in candidates])
        sol = master.solve()
        if sol.mass > before+1e-6:
            raise RuntimeError('Adding columns unexpectedly increased optimal mass')
        emit({'operation':'b_after_c', 'cycle':cycle, 'added_columns':count,
              'mass_before_same_rows':before, 'mass_improvement_same_rows':before-sol.mass,
              **summary(sol)})
        checkpoint(args.out, master, sol, history, config)
    # Stronger held-out search; if it finds a violation, feed it back into b.
    # Each repair uses a new seed; there is always a final check after the last solve.
    for j in range(args.validation_rounds+1):
        witnesses, validation = separate(master, sol, args.random_seed+999999+j,
                                         max(14, args.separation_power), local_starts=32)
        emit({'operation':'independent_validation', 'round':j, **validation})
        np.savez_compressed(args.out/'validation_witnesses.npz', poses=witnesses)
        if not len(witnesses) or j == args.validation_rounds:
            break
        master.add_rows(witnesses)
        sol = master.solve()
        emit({'operation':'b_validation_repair', 'round':j, **summary(sol)})
        checkpoint(args.out, master, sol, history, config)
    emit({'operation':'timing', 'geometry_seconds':master.geometry_seconds,
          'tiny_entries_dropped':master.tiny_entries_dropped,
          'coverage_entries_evaluated_for_master':master.entries_evaluated})
    checkpoint(args.out, master, sol, history, config)

    if args.global_check:
        from global_separation import separate_global
        for j in range(args.global_rounds+1):
            witnesses, report = separate_global(master,sol,budget=args.screen_budget)
            emit({**report,'round':j})
            checkpoint(args.out,master,sol,history,config)
            np.savez_compressed(args.out/'global_witnesses.npz',poses=witnesses)
            if report['status']=='SCREENED_ALL_NET_CENTERS':
                break
            if not len(witnesses) or j==args.global_rounds:
                raise RuntimeError('Global screening incomplete; candidate and witnesses saved')
            added=master.add_rows(witnesses)
            if not added:
                raise RuntimeError('Global witness cannot add a fresh constraint')
            sol=master.solve()
            emit({'operation':'b_global_repair','round':j,**summary(sol)})
            checkpoint(args.out,master,sol,history,config)
    plot_source=args.out/'candidate.json'
    if args.certify:
        from certify import certify
        certify(plot_source.resolve(),(args.out/'certificate').resolve())
        plot_source=args.out/'certificate/certified_candidate.json'
    if args.plot:
        from plot_solution import plot_solution
        plot_solution(plot_source,args.out/'solution.png')


if __name__ == '__main__':
    main()
