"""Hyperbolic-radius layout: radius encodes degree via r = 1 - tanh(degree/beta),
so isolated (degree-0) nodes land exactly on the boundary (r=1) automatically
-- no manual fallback-shell placement needed, unlike every giant-component-only
method elsewhere in this package. Inspired by "Hyperbolic Embedding of
Multilayer Networks" (arXiv:2505.20378), adapted to 3D and layer-aware for
our data: polar angle is banded by layer3 (three latitude bands), azimuthal
angle reuses the fa2_1 2D layout (a real structural signal), and radius
comes from degree via the paper's tanh formula.
"""

from __future__ import annotations

import math

import numpy as np

import common as c
from method_centrality import _get_or_compute_fa2_xy

LAYER_POLAR_ANGLE = {
    "protein": math.pi / 6,  # 30 degrees from north pole
    "bridge": math.pi / 2,  # 90 degrees (equator)
    "metabolite": 5 * math.pi / 6,  # 150 degrees
}


def run_hyp_1(G, node_layer, nodes_path):
    xy, recomputed_xy = _get_or_compute_fa2_xy(G, nodes_path)

    degrees = dict(G.degree())
    nonzero_degrees = [d for d in degrees.values() if d > 0]
    beta = (sum(nonzero_degrees) / len(nonzero_degrees)) if nonzero_degrees else 1.0

    x_vals = [p[0] for p in xy.values()]
    y_vals = [p[1] for p in xy.values()]
    cx, cy = sum(x_vals) / len(x_vals), sum(y_vals) / len(y_vals)

    coords = {}
    for n in G.nodes():
        x, y = xy[n]
        theta = math.atan2(y - cy, x - cx)
        phi = LAYER_POLAR_ANGLE[node_layer[n]]
        r = (1 - math.tanh(degrees[n] / beta)) * c.HYP_RADIUS_SCALE
        coords[n] = np.array(
            [
                r * math.sin(phi) * math.cos(theta),
                r * math.sin(phi) * math.sin(theta),
                r * math.cos(phi),
            ]
        )

    fallback_notes = (
        "isolated (degree-0) nodes land exactly on the boundary (r=HYP_RADIUS_SCALE) "
        "automatically via r=(1-tanh(0/beta))=1 -- no separate fallback placement needed"
    )
    if recomputed_xy:
        fallback_notes += (
            "; x_fa2_1/y_fa2_1 not found in nodes.tsv, recomputed fa2_1-equivalent xy "
            "fresh for azimuth"
        )

    meta = dict(
        method=(
            "Hyperbolic-radius layout: r=1-tanh(degree/beta) (Poincare-ball-inspired, "
            "per arXiv:2505.20378), theta from fa2_1 azimuth, phi banded by layer3"
        ),
        library_call="r=(1-tanh(degree/beta))*HYP_RADIUS_SCALE; theta=atan2(fa2_1 xy); phi=layer3 band",
        params={
            "beta": round(beta, 4),
            "radius_scale": c.HYP_RADIUS_SCALE,
            "layer_polar_angle": {k: round(v, 4) for k, v in LAYER_POLAR_ANGLE.items()},
        },
        weighting_desc="none",
        fallback_notes=fallback_notes,
    )
    return coords, meta
