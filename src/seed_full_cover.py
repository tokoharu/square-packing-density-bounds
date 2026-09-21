"""Trivial seed: one interior square covering the whole container, weighted to
prove N unit squares cannot fit. No numerics beyond Fraction; no engine.py/b/c
search involved. Meant as the easiest possible starting incumbent for advance.py.
"""
import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path

# Same constants certify.py's angular-net/smoothing safety check uses.
EPSILON = F(1, 20000)
D = F(83, 40000)


def lower_area(B, delta):
    """Any placed B-square loses at most this much area to the [delta, L-delta]^2
    interior rectangle's four boundary strips (see ADVANCE_ja.md 'fallback' bound)."""
    return B*B - 6*B*delta


def required_mass(L, B, rhs, delta):
    """Smallest weight on the single full-cover rectangle meeting rhs coverage
    against every placement, rounded up to a 1e-12 grid so JSON/float64 can only
    round the density up further, never down."""
    area = (L-2*delta)**2
    la = lower_area(B, delta)
    if la <= 0:
        raise ValueError('delta too large: fallback area lower bound is not positive')
    needed = rhs*area/la
    unit = 10**12
    return F((needed.numerator*unit+needed.denominator-1)//needed.denominator, unit)


def search_max_L(n, B, rhs, delta, buffer, step):
    """Largest multiple of step with required_mass(L) <= n - buffer.
    Only ever used to pick a comfortable default; --L overrides this."""
    target = F(n)-buffer
    if target <= 0:
        raise ValueError('buffer must be smaller than N')
    la = lower_area(B, delta)
    est = 2*float(delta)+math.sqrt(max(float(target)*float(la)/float(rhs), 0.))
    L = (F(est)/step).__floor__()*step
    while required_mass(L, B, rhs, delta) > target:
        L -= step
        if L <= 2*delta:
            raise ValueError('No feasible L found; loosen --buffer, shrink --delta, or shrink --step')
    return L


def build(n, L, B, rhs, delta):
    if not 0 < B < 1:
        raise ValueError('Require 0 < B < 1')
    if not B*(1+D)+3*EPSILON < 1:
        raise ValueError('B fails certify.py angular-net/smoothing safety condition')
    if not L*L > 2*B*B:
        raise ValueError('Require L > sqrt(2)*B')
    if delta <= EPSILON:
        raise ValueError('delta must exceed certify.py epsilon (1/20000)')
    mass = required_mass(L, B, rhs, delta)
    if not mass < n:
        raise ValueError(f'mass={float(mass)} >= N={n}; raise --buffer, lower --L, or shrink --delta')
    area = (L-2*delta)**2
    coverage = mass*lower_area(B, delta)/area
    if coverage < rhs:
        raise ValueError('Rounded weight lost the coverage bound (should not happen)')
    weight = float(mass)
    # Re-check with the exact decimal string that will actually reach JSON and
    # certify.py's parse_float=Fraction reader, since that's what gets checked,
    # not our higher-precision intermediate Fraction.
    served_mass = F(repr(weight))
    if not served_mass < n or served_mass*lower_area(B, delta)/area < rhs:
        raise ValueError('float64 serialization broke the exact safety margin; shrink --delta or --L')
    rect = [float(delta), float(delta), float(L-delta), float(L-delta)]
    return {
        'L': float(L), 'B': float(B), 'rhs': float(rhs),
        'rectangles': [rect], 'weights': [weight],
        'globally_verified': False, 'status': 'TRIVIAL_FULL_COVER_SEED',
        'seed': {'n': n, 'delta': str(delta), 'mass_exact': str(mass),
                 'coverage_lower_bound_exact': str(coverage),
                 'construction': 'single_interior_square_uniform_density',
                 'sqrt_n': math.sqrt(n)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=29)
    ap.add_argument('--L', type=str, help='Exact L (e.g. "5.32" or "133/25"); default: search below sqrt(n)')
    ap.add_argument('--B', type=str, default='9977/10000')
    ap.add_argument('--rhs', type=str, default='1001/1000')
    ap.add_argument('--delta', type=str, default='1/10000', help='Wall margin of the covering rectangle')
    ap.add_argument('--buffer', type=str, default='1/2', help='Headroom below N when auto-searching L')
    ap.add_argument('--step', type=str, default='1/10000', help='Grid step for auto-searching L')
    ap.add_argument('--out', type=Path, required=True, help='Output directory (candidate.json goes here)')
    ap.add_argument('--certify', action='store_true', help='Also run certify.py (needs g++) into --out/certificate')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    if args.n < 1:
        ap.error('--n must be positive')
    B, rhs, delta, buffer, step = (F(args.B), F(args.rhs), F(args.delta), F(args.buffer), F(args.step))
    L = F(args.L) if args.L else search_max_L(args.n, B, rhs, delta, buffer, step)
    data = build(args.n, L, B, rhs, delta)
    args.out.mkdir(parents=True, exist_ok=True)
    candidate = args.out/'candidate.json'
    candidate.write_text(json.dumps(data, indent=2))
    report = {'candidate': str(candidate), 'n': args.n, 'L': data['L'],
              'sqrt_n': data['seed']['sqrt_n'], 'gap_to_sqrt_n': data['seed']['sqrt_n']-data['L'],
              'mass_exact': data['seed']['mass_exact'], 'mass_float': sum(data['weights']),
              'coverage_lower_bound_exact': data['seed']['coverage_lower_bound_exact']}
    if args.certify:
        from certify import certify
        meta = certify(candidate.resolve(), (args.out/'certificate').resolve(), args.workers)
        report['certificate'] = meta
        report['certified_candidate'] = str(args.out/'certificate/certified_candidate.json')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
