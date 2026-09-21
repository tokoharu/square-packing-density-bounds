"""Local-translation proposals: relocate a positive-weight rectangle without
resizing it. Parents are retained; the LP decides how much weight moves."""
import numpy as np
from scipy.optimize import minimize
from pricing import scores


def propose_translations(master, solution, max_parents=4, local_starts=4, seed=0):
    if max_parents < 1 or local_starts < 1:
        raise ValueError('Invalid refinement budget')
    rng = np.random.default_rng(seed)
    records = []
    for i in np.flatnonzero(solution.weights > 0):
        r = master.rectangles[i]
        size = r[2:]-r[:2]
        lo, hi = np.zeros(2), master.L-size
        parent_score = float(scores(master.L, master.B, [r], master.poses, solution.dual)[0])
        if np.all(hi <= lo):
            continue  # spans the container on both axes; nowhere to move

        def rectangle_at(origin, size=size, lo=lo, hi=hi):
            o = np.clip(origin, lo, hi)
            return np.concatenate([o, o+size])

        def objective(origin, rectangle_at=rectangle_at):
            return -float(scores(master.L, master.B, [rectangle_at(origin)], master.poses, solution.dual)[0])

        starts = [r[:2]]+[np.clip(r[:2]+rng.uniform(-1, 1, 2)*(hi-lo), lo, hi)
                          for _ in range(local_starts-1)]
        best_origin, best_score = r[:2], parent_score
        for start in starts:
            opt = minimize(objective, start, method='Nelder-Mead', bounds=list(zip(lo, hi)),
                           options={'maxiter': 140, 'xatol': 1e-6, 'fatol': 1e-8})
            if -opt.fun > best_score:
                best_score, best_origin = -opt.fun, opt.x
        moved = rectangle_at(best_origin)
        advantage = best_score-parent_score
        records.append({'parent_index': int(i), 'parent_weight': float(solution.weights[i]),
                        'parent_rectangle': r.tolist(), 'rectangle': moved.tolist(),
                        'offset': (best_origin-r[:2]).tolist(), 'parent_score': parent_score,
                        'score': best_score, 'advantage': advantage,
                        'priority': float(solution.weights[i]*max(0., advantage))})
    records.sort(key=lambda r: -r['priority'])
    selected = [r for r in records if r['advantage'] > 1e-6][:max_parents]
    return selected, records
