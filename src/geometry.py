"""Normalized D4 rectangle bases; all integration here is numerical, not proof."""
import numpy as np
from fast_geometry import matrix


class Geometry:
    def __init__(self, L, B, rectangles):
        self.L, self.B = float(L), float(B)
        if not np.isfinite([L, B]).all() or not L > np.sqrt(2)*B > 0:
            raise ValueError('Require L > sqrt(2)*B > 0')
        self.rectangles = np.asarray(rectangles, dtype=float).reshape(-1, 4)
        r = self.rectangles
        if (not np.isfinite(r).all() or np.any(r[:, :2] < 0)
                or np.any(r[:, 2:] > L) or np.any(r[:, 2:] <= r[:, :2])):
            raise ValueError('Rectangles must have positive area inside [0,L]^2')
        full = []
        for x0, y0, x1, y1 in r:
            for swap in [False, True]:
                a, b, c, d = (y0, x0, y1, x1) if swap else (x0, y0, x1, y1)
                for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    u, v = (a, c) if sx == 1 else (L-c, L-a)
                    w, z = (b, d) if sy == 1 else (L-d, L-b)
                    full.append([u, w, v, z])
        self.full = np.asarray(full, dtype=float).reshape(-1, 4)
        self.areas = np.prod(self.full[:, 2:]-self.full[:, :2], axis=1)

    def matrix(self, poses):
        p = np.asarray(poses, dtype=float).reshape(-1, 3)
        if (not np.isfinite(p).all() or np.any(np.abs(p[:, :2]) > 1)
                or np.any(p[:, 2] < 0) or np.any(p[:, 2] > 1)):
            raise ValueError('Pose=(normalized cx,cy,theta); xy in [-1,1], theta in [0,1]')
        return matrix(p, self.L, self.B, self.full, self.areas,
                      np.arange(len(self.rectangles)))

    def axis_poses(self):
        coords = np.unique(self.full[:, [0, 2]].ravel())
        centers = np.r_[self.L/2, self.L-self.B/2, coords-self.B/2, coords+self.B/2]
        centers = centers[(centers >= self.L/2) & (centers <= self.L-self.B/2)]
        p = np.unique((centers-self.L/2)/((self.L-self.B)/2))
        xx, yy = np.meshgrid(p, p, indexing='ij')
        return np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]


def rectangle_key(r, L):
    """Canonical D4 key, for duplicate rejection only; never round actual geometry."""
    full = Geometry(L, min(1., L/2), [r]).full
    return min(tuple(np.round(x, 12)) for x in full)
