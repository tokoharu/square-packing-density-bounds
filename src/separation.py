"""Numerical counterexample search: complete axis event grid + sampled rotations."""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from geometry import Geometry


def separate(master, solution, seed=0, samples_power=12, local_starts=12, max_rows=64):
    start = perf_counter()
    if samples_power < 1 or local_starts < 0 or max_rows < 1:
        raise ValueError('Invalid separation budget')
    # Retain all positive coefficients, including small corrections.
    active = solution.weights > 0
    model = Geometry(master.L, master.B, master.rectangles[active])
    weight = solution.weights[active]
    axis = model.axis_poses()
    samples = qmc.Sobol(3, scramble=True, seed=seed).random_base2(samples_power)
    # Include exactly 45 degrees, which random interior sampling would omit.
    edge = samples[:min(256, len(samples))].copy(); edge[:, 2] = 1
    poses = np.vstack([axis, samples, edge])
    values = model.matrix(poses)@weight
    best = np.argsort(values)[:max(64, local_starts)]
    candidates = list(poses[best])
    def objective(p):
        return float(model.matrix(p)[0]@weight)
    for i in best[:local_starts]:
        opt = minimize(objective, poses[i], method='Nelder-Mead', bounds=[(0., 1.)]*3,
                       options={'maxiter':300, 'xatol':1e-7, 'fatol':1e-9})
        # Bounded Nelder-Mead can return x marginally outside its declared
        # bounds (floating-point overshoot near the boundary); re-clip so
        # candidates always satisfy master.add_rows' pose validation.
        candidates.append(np.clip(opt.x, 0., 1.))
    # Keep all sampled violations eligible, even when max_rows > 64.
    bad_samples = poses[values < master.rhs-2e-7]
    candidates = np.vstack([np.asarray(candidates), bad_samples])
    cv = model.matrix(candidates)@weight
    fresh, seen = [], set(master.pose_keys)
    for i in np.argsort(cv):
        if cv[i] >= master.rhs-2e-7:
            break
        key = tuple(np.round(candidates[i], 12))
        if key not in seen:
            fresh.append(candidates[i]); seen.add(key)
            if len(fresh) == max_rows:
                break
    minimum = float(min(values.min(), cv.min()))
    return np.asarray(fresh).reshape(-1, 3), {
        'status': 'NUMERICAL_VIOLATIONS' if minimum < master.rhs-2e-7 else 'NO_VIOLATION_FOUND',
        'minimum_observed': minimum, 'axis_minimum': float(values[:len(axis)].min()),
        'axis_events': len(axis), 'sampled_poses': len(poses),
        'new_rows': len(fresh), 'seconds': perf_counter()-start,
        'globally_verified': False}
