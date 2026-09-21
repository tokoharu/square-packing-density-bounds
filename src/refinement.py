"""Long-axis bisection proposals. Parents are retained to preserve the old cone."""
import numpy as np
from pricing import scores


def bisect_long(rectangle,axis=None):
    """Bisect `rectangle` at the midpoint of `axis` (0=x, 1=y). axis=None (default) picks
    the longer side, as before. An explicit axis lets a caller request the shorter side
    instead; the short dimension is otherwise left untouched either way."""
    r=np.asarray(rectangle,float)
    lengths=r[2:]-r[:2]
    if r.shape!=(4,) or not np.isfinite(r).all() or np.any(lengths<=0):
        raise ValueError('Invalid rectangle')
    if axis is None:
        axis=int(np.argmax(lengths))
    elif axis not in (0,1):
        raise ValueError('axis must be 0, 1, or None')
    middle=r[axis]+lengths[axis]/2
    if not r[axis]<middle<r[axis+2]:raise ValueError('Subdivision below float resolution')
    first=r.copy();second=r.copy()
    first[axis+2]=middle;second[axis]=middle
    fractions=np.array([(first[axis+2]-first[axis])/lengths[axis],
                        (second[axis+2]-second[axis])/lengths[axis]])
    return np.array([first,second]),fractions,axis


def propose_splits(master,solution,max_parents=4,min_aspect=2.,split_axis='long',short_floor=1e-3):
    """split_axis='long' (default) always bisects each candidate's longer side, as before.
    split_axis='short' bisects the shorter side instead, but only for candidates whose
    shorter side is still >= short_floor; below that it falls back to the long side for that
    candidate. Each candidate is a fresh, independent one-off choice per call (this function
    is not recursive), so unlike repeatedly re-splitting a box across many rounds, there is no
    runaway shrinkage: a rectangle that keeps getting proposed across cycles is re-evaluated
    from scratch every time, and once its shorter side drops under short_floor this rule stops
    thinning it further and defers to the long side."""
    if max_parents<1 or min_aspect<1:raise ValueError('Invalid refinement budget')
    if split_axis not in ('long','short'):raise ValueError('split_axis must be "long" or "short"')
    records=[]
    for i in np.flatnonzero(solution.weights>0):
        r=master.rectangles[i];lengths=r[2:]-r[:2]
        if max(lengths)/min(lengths)<min_aspect:continue
        forced_axis=int(np.argmin(lengths)) if split_axis=='short' and min(lengths)>=short_floor else None
        children,fractions,axis=bisect_long(r,forced_axis)
        sc=scores(master.L,master.B,children,master.poses,solution.dual)
        parent_score=float(scores(master.L,master.B,[r],master.poses,solution.dual)[0])
        advantage=float(max(sc)-1)
        records.append({'parent_index':int(i),'parent_weight':float(solution.weights[i]),
                        'parent_rectangle':r.tolist(),'axis':'x' if axis==0 else 'y',
                        'children':children.tolist(),'fractions':fractions.tolist(),
                        'child_scores':sc.tolist(),'parent_score':parent_score,
                        'identity_error':float(fractions@sc-parent_score),
                        'priority':float(solution.weights[i]*max(0.,advantage))})
    records.sort(key=lambda r:-r['priority'])
    selected=[r for r in records if max(r['child_scores'])>1+1e-6][:max_parents]
    return selected,records
