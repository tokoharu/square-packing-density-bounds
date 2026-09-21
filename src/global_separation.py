"""Complete-center numerical screening on the proof's 201-angle net.

Only the separate outward-rounded verifier may promote a candidate to VERIFIED.
"""
from time import perf_counter
import numpy as np
from geometry import Geometry
from net_screen import verify_angle


def net_pose(r, cx, cy, L, B):
    theta = 2*np.arctan(83*r/40000)
    extent = (L-B*(np.cos(theta)+np.sin(theta)))/2
    u, v = (np.array([cx,cy])-L/2)/extent
    angle = theta/(np.pi/4)
    # Last net angle is just above pi/4: reflect across x=y.
    if angle > 1:
        u, v, angle = v, u, 2-angle
    return np.clip([u,v,angle], [0,0,0], [1,1,1])


def separate_global(master, solution, target=1.0005, budget=1000000, max_rows=24):
    if not 1.0001 < target < master.rhs-1e-6 or budget < 1 or max_rows < 1:
        raise ValueError('Require 1.0001 < screening target < rhs; positive budgets')
    start = perf_counter()
    ids = solution.weights > 0
    model = Geometry(master.L, master.B, master.rectangles[ids])
    weights = solution.weights[ids]
    axis = model.axis_poses()
    values = model.matrix(axis)@weights
    witnesses = []
    for i in np.argsort(values):
        if values[i] >= target*(1+1e-7) or len(witnesses) >= max_rows:
            break
        witnesses.append(axis[i])
    rho = np.repeat(weights/8,8)/model.areas/target
    cases, unknown, nodes = [], [], 0
    # Scan all directions, unless enough counterexamples already justify a re-solve.
    for r in range(1,201):
        if len(witnesses) >= max_rows:
            break
        t = 83*r/40000
        c, s = (1-t*t)/(1+t*t), 2*t/(1+t*t)
        result = verify_angle(c,s,master.B,master.L,model.full,rho,budget)
        status, visited, leaves, minimum, witness, pending, lower = result
        nodes += visited
        cases.append({'r':r,'status':int(status),'nodes':int(visited)})
        if status == 0:
            witnesses.append(net_pose(r,*witness[:2],master.L,master.B))
        elif status == -1:
            unknown.append(r)
    poses = np.asarray(witnesses).reshape(-1,3)
    # Independently re-evaluate returned poses in the LP geometry before adding.
    checked = model.matrix(poses)@weights if len(poses) else np.array([])
    poses = poses[checked < master.rhs-2e-7]
    complete = len(cases)==200 and not unknown and not witnesses
    return poses, {'operation':'global_separation','status':
                  'SCREENED_ALL_NET_CENTERS' if complete else
                  ('COUNTEREXAMPLES' if len(poses) else 'UNRESOLVED'),
                  'angles_checked':len(cases)+1,'unknown_angles':unknown,
                  'axis_minimum':float(values.min()),'new_rows':len(poses),
                  'nodes':int(nodes),'seconds':perf_counter()-start,
                  'screening_target':target,'globally_verified':False,'angles':cases}
