#!/usr/bin/env python3
"""push.py -- one command to drive tokoharu's rectangle-density pipeline.

The tools work, but three things have to be arranged by hand before they do:

  1. `--poses` is not shipped and depends on L, so every new side needs its own
     grid of admissible placements built from `global_separation.net_pose`.
  2. `advance.py` only accepts a *certified* incumbent, so an `engine.py` result
     has to go through `certify.py` before it can be advanced.
  3. `engine.py --global-check` fails outright when repair does not converge
     inside `--global-rounds`, which usually means the step was too ambitious
     rather than that the side is out of reach.

This wraps all three. Give it an n and either a starting certificate or nothing,
and it climbs L as far as it can, certifying every accepted step.

  python3 src/push.py --n 29 --from certificates/cert_n29_L571 \
      --target 5.73 --out runs/n29
  python3 src/push.py --n 53 --target 7.40 --out runs/n53   # starts from a seed

Each accepted rung leaves a full certificate directory, so the run can be
stopped at any point and what it has already proved stays proved.
"""
import argparse, json, math, os, shutil, subprocess, sys, time
from fractions import Fraction as F
from pathlib import Path

SRC = Path(__file__).resolve().parent


LOGFILE = None


def log(**kw):
    """Append one JSON record to stdout and, once --out is known, to a file.

    Relying on the caller to redirect stdout loses the whole run record when the
    redirect is forgotten -- which is how the reason for one n=29 rung halving
    its step went missing.  The file lives in the run directory, so it survives
    whatever the shell did or did not do.
    """
    kw['t'] = round(time.time() - START, 1)
    line = json.dumps(kw)
    print(line, flush=True)
    if LOGFILE is not None:
        with open(LOGFILE, 'a') as fh:
            fh.write(line + '\n')


def run(cmd, cwd=SRC, quiet=True):
    """Run a pipeline step; return (ok, tail_of_output)."""
    r = subprocess.run([sys.executable] + cmd, cwd=cwd,
                       capture_output=True, text=True)
    out = (r.stdout or '') + (r.stderr or '')
    return r.returncode == 0, out[-4000:]


def make_poses(L, B, out, angle_step=5, grid=18):
    """Admissible centres on a grid, at every `angle_step`-th net angle.

    Poses are the normalised triples geometry.matrix indexes: the centre
    offset from the container's middle, divided by how far it may travel at
    that angle, so xy runs over [-1, 1]; and the angle as a fraction of pi/4,
    over [0, 1]. (global_separation.net_pose computes the same thing but then
    clips xy to [0, 1], because the separation oracle only ever hands back
    first-quadrant witnesses -- that clip would collapse half of a full grid
    onto the axes.)
    """
    import numpy as np
    rows = []
    for r in range(0, 201, angle_step):
        t = 2 * math.atan(r * 83 / 40000)
        c, s = math.cos(t), math.sin(t)
        extent = (L - B * (c + s)) / 2
        if extent <= 0:
            continue
        angle = t / (math.pi / 4)
        for i in range(grid + 1):
            u = -1 + 2 * i / grid
            for j in range(grid + 1):
                v = -1 + 2 * j / grid
                if angle > 1:            # last node sits just past pi/4
                    rows.append((v, u, 2 - angle))
                else:
                    rows.append((u, v, angle))
    P = np.asarray(rows, dtype=float).reshape(-1, 3)
    np.savez_compressed(out, poses=P)
    return len(P)


def rung_tag(L):
    """Directory-safe name for a rung at side L.

    L is a Fraction, and str() would render 5.7125 as "457/80"; the slash would
    silently nest the run directory one level deeper and break every path built
    from it afterwards.
    """
    return f'{float(L):.8f}'.rstrip('0').rstrip('.').replace('.', '_')


def axis_poses_representable(L, B, rectangles):
    """Would geometry.axis_poses stay inside the box geometry.matrix accepts?"""
    sys.path.insert(0, str(SRC))
    import numpy as np
    from geometry import Geometry
    try:
        g = Geometry(L, B, rectangles)
    except Exception:
        return False
    coords = np.unique(g.full[:, [0, 2]].ravel())
    centres = np.r_[L / 2, L - B / 2, coords - B / 2, coords + B / 2]
    centres = centres[(centres >= L / 2) & (centres <= L - B / 2)]
    if not len(centres):
        return False
    return float(((centres - L / 2) / ((L - B) / 2)).max()) <= 1.0


def next_step(step, initial, streak, ok, widen_after=2, floor=None):
    """How wide the next rung should be, and the streak that goes with it.

    Halving on failure without ever widening again makes the step a ratchet: the
    increment ends up set by the total number of failures, however many rungs
    have succeeded since.  A transient failure -- a repair that ran out of rounds
    under load, say -- then costs the rest of the run twice the rungs for the
    same distance.  Widen back after a couple of clean rungs, never past where
    the run started, and never narrow below `floor`: a ladder that keeps making
    progress should not be able to grind its own increment to nothing.
    """
    if not ok:
        narrowed = step / 2
        if floor is not None and narrowed < floor:
            narrowed = floor
        return narrowed, 0
    streak += 1
    if streak >= widen_after and step < initial:
        return min(step * 2, initial), 0
    return step, streak


def certified(d):
    """Is this directory a finished certificate we can advance from?"""
    c = Path(d) / 'certified_candidate.json'
    if not c.exists():
        return False
    try:
        j = json.loads(c.read_text())
        s = json.loads((Path(d) / 'verification_summary.json').read_text())
        return j.get('globally_verified') and s.get('status') == 'VERIFIED'
    except Exception:
        return False


def seed(n, work):
    """Weakest possible starting certificate: one rectangle covering the box."""
    out = work / 'seed'
    ok, tail = run(['seed_full_cover.py', '--n', str(n), '--out', str(out)])
    if not ok:
        log(step='seed', status='FAILED', tail=tail[-500:])
        return None
    L = json.loads((out / 'candidate.json').read_text())['L']
    log(step='seed', status='OK', L=L)
    return out / 'candidate.json'


# What geometry.matrix raises when an axis pose lands a rounding step outside
# the box it accepts. attempt()'s pre-check predicts this from the rung's
# starting rectangles, but engine.py's own column generation can introduce
# axis poses the pre-check never saw -- so a run can still hit this deep
# inside --global-check. Recognising the message here means that case is
# still reported as AXIS_POSES_OUT_OF_RANGE instead of a generic engine
# failure, regardless of whether the pre-check predicted it.
AXIS_POSE_ERROR = 'Pose=(normalized cx,cy,theta); xy in [-1,1], theta in [0,1]'


def why(tail):
    """The most specific line of a failed step's output.

    The reason classification above names the *kind* of failure; this keeps the
    message itself.  Without it a rung that dies after engine.py has already
    written its whole output directory reports a bare ENGINE_FAILED, which says
    nothing about whether the side is out of reach or the run hit a bug.  The
    last non-empty line is where a Python traceback puts its exception, and
    where a clean early exit puts its message.
    """
    lines = [l.rstrip() for l in tail.splitlines() if l.strip()]
    return lines[-1][:300] if lines else ''



def search(candidate, poses, out, cycles, rounds):
    """engine.py: grow the basis until every net angle screens clean."""
    ok, tail = run(['engine.py', '--seed', str(candidate), '--poses', str(poses),
                    '--out', str(out), '--cycles', str(cycles), '--b-rounds', '3',
                    '--global-check', '--global-rounds', str(rounds)])
    if not ok:
        if AXIS_POSE_ERROR in tail:
            reason = 'AXIS_POSES_OUT_OF_RANGE'
        elif 'screening incomplete' in tail:
            reason = 'REPAIR_DID_NOT_CONVERGE'
        else:
            reason = 'ENGINE_FAILED'
        return False, (reason, why(tail))
    return True, ('SCREENED', '')


def certify(candidate, out, workers):
    """certify.py: exact interval verification of all 201 angles."""
    ok, tail = run(['certify.py', str(candidate), '--out', str(out),
                    '--workers', str(workers)])
    return (ok and certified(out)), tail


def attempt(n, L, parent_cert, work, rung, cycles, rounds, workers):
    """One rung of the ladder: build a certificate for this n at this L."""
    d = work / f'L{rung_tag(L)}'
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

    # start from the parent's density, re-labelled at the new side
    src = json.loads((parent_cert / 'certified_candidate.json').read_text())
    src['L'] = float(L)
    src.pop('certificate', None)
    src['globally_verified'] = False
    src['status'] = 'PROPOSED'
    cand = d / 'proposal.json'
    cand.write_text(json.dumps(src))

    npz = d / 'poses.npz'
    make_poses(float(L), float(src['B']), npz)

    # geometry.axis_poses divides centres by (L-B)/2 and geometry.matrix then
    # rejects anything above 1. The centre L-B/2 normalises to exactly 1, so
    # for some L it lands a rounding step above and the engine dies on its own
    # generator before doing any work. Skip those sides rather than spend the
    # ladder's step budget halving into them.
    if not axis_poses_representable(float(L), float(src['B']),
                                    [tuple(r) for r in src['rectangles']]):
        return None, ('AXIS_POSES_OUT_OF_RANGE', '')

    ok, (reason, detail) = search(cand, npz, d / 'search', cycles, rounds)
    if not ok:
        return None, (reason, detail)
    ok, tail = certify(d / 'search' / 'candidate.json', d / 'cert', workers)
    if not ok:
        return None, ('CERTIFY_FAILED', why(tail))
    m = json.loads((d / 'cert' / 'certificate_metadata.json').read_text())
    if F(m['mass_exact']) >= n:
        return None, ('MASS_NOT_BELOW_N',
                      f"mass {float(F(m['mass_exact'])):.9f} against budget {n}")
    return d / 'cert', float(m['mass_decimal'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--from', dest='start', type=Path,
                    help='an existing certificate directory to climb from')
    ap.add_argument('--target', type=F, help='stop once L reaches this')
    ap.add_argument('--step', type=F, default=F(1, 200))
    ap.add_argument('--min-step', type=F, default=F(1, 12800),
                    help='never narrow the increment below this')
    ap.add_argument('--give-up-after', type=int, default=4,
                    help='stop after this many consecutive failed rungs')
    ap.add_argument('--out', type=Path, required=True,
                    help='working directory; one subdirectory per accepted rung')
    ap.add_argument('--cycles', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=60,
                    help='global repair rounds; raise before lowering the step')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--max-rungs', type=int, default=40)
    ap.add_argument('--widen-after', type=int, default=2,
                    help='double the step again after this many clean rungs')
    a = ap.parse_args()

    work = a.out.resolve()
    work.mkdir(parents=True, exist_ok=True)
    global LOGFILE
    LOGFILE = work / 'push.jsonl'
    log(step='launch', n=a.n, target=float(a.target) if a.target else None,
        step_size=str(a.step), min_step=str(a.min_step), rounds=a.rounds,
        give_up_after=a.give_up_after, widen_after=a.widen_after,
        source=str(a.start) if a.start else 'seed', log=str(LOGFILE))

    if a.start and certified(a.start):
        best = Path(a.start).resolve()
        L = F(json.loads((best / 'certificate_metadata.json').read_text())['L'])
        log(step='start', source='certificate', L=float(L))
    else:
        c = seed(a.n, work)
        if not c:
            return 1
        cand = json.loads(c.read_text())
        L0 = F(str(cand['L']))
        npz = work / 'seed_poses.npz'
        make_poses(float(L0), float(cand['B']), npz)
        # Same danger zone attempt() guards against: a fresh seed for a large n
        # can itself land at an axis pose geometry.matrix rejects, which would
        # otherwise kill the very first search before it does any work.
        if not axis_poses_representable(float(L0), float(cand['B']),
                                        [tuple(r) for r in cand['rectangles']]):
            log(step='start', status='AXIS_POSES_OUT_OF_RANGE')
            return 1
        ok, (reason, detail) = search(c, npz, work / 'seed_search',
                                      a.cycles, a.rounds)
        if not ok:
            log(step='start', status=reason, detail=detail)
            return 1
        ok, _ = certify(work / 'seed_search' / 'candidate.json',
                        work / 'seed_cert', a.workers)
        if not ok:
            log(step='start', status='CERTIFY_FAILED')
            return 1
        # certify.py deliberately doesn't check this; attempt() does for every
        # later rung, so the seed certificate needs the same guard.
        m = json.loads((work / 'seed_cert' / 'certificate_metadata.json').read_text())
        if F(m['mass_exact']) >= a.n:
            log(step='start', status='MASS_NOT_BELOW_N')
            return 1
        best, L = work / 'seed_cert', L0
        log(step='start', source='seed', L=float(L))

    step = a.step
    streak = 0
    consecutive = 0
    for rung in range(a.max_rungs):
        if a.target is not None and L >= a.target:
            log(step='done', status='TARGET_REACHED', best_L=float(L))
            break
        nxt = L + step
        if a.target is not None and nxt > a.target:
            nxt = a.target
        got, info = attempt(a.n, nxt, best, work, rung, a.cycles, a.rounds, a.workers)
        if got:
            best, L = got, nxt
            log(step='rung', L=float(nxt), status='ACCEPTED', mass=info,
                budget=a.n, increment=str(step))
            consecutive = 0
            was = step
            step, streak = next_step(step, a.step, streak, True, a.widen_after,
                                     a.min_step)
            if step != was:
                log(step='widen', increment=str(step))
            continue
        status, detail = info
        consecutive += 1
        log(step='rung', L=float(nxt), status=status, detail=detail,
            increment=str(step), consecutive=consecutive)
        step, streak = next_step(step, a.step, streak, False, a.widen_after,
                                 a.min_step)
        # Stop on repeated failure at one place, not on a tally of failures over
        # the whole run.  Counting every failure ends a ladder that is still
        # making progress -- n=26 and n=39 each failed a rung to --global-rounds
        # being too low, then climbed several more once it was raised.
        if consecutive >= a.give_up_after:
            log(step='done', status='STUCK', best_L=float(L),
                consecutive=consecutive)
            break
    else:
        log(step='done', status='MAX_RUNGS', best_L=float(L))

    log(step='final', n=a.n, best_L=float(L), certificate=str(best))
    return 0


START = time.time()
if __name__ == '__main__':
    raise SystemExit(main())
