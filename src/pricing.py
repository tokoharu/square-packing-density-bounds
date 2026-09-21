"""c: free-position, multiscale rectangle pricing from the current LP dual.

Search is heuristic. Scores use polygon intersections, not a raster surrogate.
Failure to find a column is NOT a certificate of pricing optimality.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from geometry import Geometry, rectangle_key


def scores(L, B, rectangles, poses, dual):
    # Zero dual rows contribute nothing. Keep EVERY strictly positive row.
    ids = np.flatnonzero(dual > 0)
    result = []
    for start in range(0, len(rectangles), 256):
        a = Geometry(L, B, rectangles[start:start+256]).matrix(poses[ids])
        result.extend(dual[ids]@a)
    return np.asarray(result)


def propose(master, solution, seed=0, samples_power=9, local_starts=2,
            max_columns=12, widths=(1/64, 1/32, 1/16, 1/8, 1/4, 1/2), margin=.001):
    start = perf_counter()
    limit = master.L/2
    if margin <= 0 or margin >= limit:
        raise ValueError('Invalid support margin')
    if not widths or any(not np.isfinite(w) or w <= 0 for w in widths):
        raise ValueError('Widths must be positive and finite')
    if samples_power < 1 or local_starts < 0 or max_columns < 1:
        raise ValueError('Invalid pricing budget')
    points = qmc.Sobol(2, scramble=True, seed=seed).random_base2(samples_power)
    # Include domain edges and corners as well as low-discrepancy interior points.
    v = np.linspace(0, 1, 17)
    points = np.vstack([points, np.c_[v, np.zeros(17)], np.c_[v, np.ones(17)],
                        np.c_[np.zeros(17), v], np.c_[np.ones(17), v]])
    proposals = []
    evaluated = 0
    for w in widths:
        for h in widths:
            if w > h or max(w, h) > limit-margin:
                continue
            size = np.array([w, h])
            lo, hi = margin+size/2, limit-size/2
            def rectangles_at(p):
                centers = lo+np.atleast_2d(p)*(hi-lo)
                return np.c_[centers-size/2, centers+size/2]
            rects = rectangles_at(points)
            values = scores(master.L, master.B, rects, master.poses, solution.dual)
            evaluated += len(rects)
            best = np.argsort(-values)[:max(2, local_starts)]
            pool = [(rects[i], float(values[i])) for i in best]
            for i in best[:local_starts]:
                def objective(p):
                    return -float(scores(master.L, master.B, rectangles_at(p),
                                         master.poses, solution.dual)[0])
                opt = minimize(objective, points[i], method='Nelder-Mead',
                               bounds=[(0., 1.), (0., 1.)],
                               options={'maxiter':140, 'xatol':1e-6, 'fatol':1e-8})
                evaluated += opt.nfev
                pool.append((rectangles_at(opt.x)[0], -float(opt.fun)))
            # Preserve two locations per scale, rather than only thin global maxima.
            seen = set()
            for r, value in sorted(pool, key=lambda x: -x[1]):
                key = rectangle_key(r, master.L)
                if key not in seen and key not in master.rect_keys and value > 1+1e-6:
                    proposals.append((r, value, (w, h)))
                    seen.add(key)
                    if len(seen) == 2:
                        break
    # First take the best from each scale, then remaining proposals.
    first, rest, used_scales = [], [], set()
    for item in sorted(proposals, key=lambda x: -x[1]):
        (rest if item[2] in used_scales else first).append(item)
        used_scales.add(item[2])
    selected, seen = [], set(master.rect_keys)
    for r, value, scale in first+rest:
        key = rectangle_key(r, master.L)
        if key in seen:
            continue
        selected.append({'rectangle': r.tolist(), 'score': value, 'scale': list(scale)})
        seen.add(key)
        if len(selected) == max_columns:
            break
    return selected, {'seconds': perf_counter()-start, 'evaluations': int(evaluated),
                      'positive_dual_rows': int(np.count_nonzero(solution.dual > 0)),
                      'best_score': max((p['score'] for p in selected), default=None),
                      'status': 'HEURISTIC_PRICING', 'columns': len(selected)}
