"""Generalized Procrustes ensemble blend: aligns and averages 2-3
complementary existing layouts into one, combining their individual
strengths instead of relying on any single method's biases (Gower,
"Generalized Procrustes Analysis", Psychometrika 1975 -- the standard
technique for combining multiple point-configurations of the same objects
into a consensus frame). Naively averaging un-aligned point clouds would be
meaningless since each layout's rotation/reflection/scale is arbitrary;
`scipy.linalg.orthogonal_procrustes` finds the optimal rotation to align one
onto another given the known node-id correspondence, then a weighted
average combines them. Blends fa23d_1 (organic local clustering, force-
directed), mds_1 (global geodesic-distance fidelity), and spectral_1
(algebraic connectivity, handles disconnected structure well) -- three
deliberately different objectives, not near-duplicates.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes

import common as c
import method_distance_embedding as m_dist
import method_forceatlas as m_fa2

BLEND_WEIGHTS = {"fa23d_1": 0.4, "mds_1": 0.3, "spectral_1": 0.3}


def _center_scale(coords_dict: dict[str, tuple], nodes: list[str]) -> np.ndarray:
    arr = np.array([coords_dict[n] for n in nodes], dtype=float)
    centroid = arr.mean(axis=0)
    arr = arr - centroid
    scale = float(np.sqrt((arr**2).sum(axis=1).mean())) or 1.0
    return arr / scale


def run_ensemble_1(G, node_layer, nodes_path):
    nodes = sorted(G.nodes(), key=c.node_key)

    fa_coords, _ = m_fa2.run_fa23d_1(G, node_layer, nodes_path)
    mds_coords, _ = m_dist.run_mds_1(G, node_layer, nodes_path)
    spectral_coords, _ = m_dist.run_spectral_1(G, node_layer, nodes_path)

    A = _center_scale(fa_coords, nodes)  # reference frame
    B = _center_scale(mds_coords, nodes)
    Cc = _center_scale(spectral_coords, nodes)

    R_b, _ = orthogonal_procrustes(B, A)
    R_c, _ = orthogonal_procrustes(Cc, A)
    B_aligned = B @ R_b
    C_aligned = Cc @ R_c

    w = BLEND_WEIGHTS
    blend = w["fa23d_1"] * A + w["mds_1"] * B_aligned + w["spectral_1"] * C_aligned
    coords = {n: blend[i] for i, n in enumerate(nodes)}

    meta = dict(
        method="Generalized Procrustes ensemble blend of fa23d_1 + mds_1 + spectral_1",
        library_call=(
            "scipy.linalg.orthogonal_procrustes(mds_1, fa23d_1) and "
            "orthogonal_procrustes(spectral_1, fa23d_1) to align onto fa23d_1's frame, "
            "then weighted average"
        ),
        params={"weights": BLEND_WEIGHTS, "seed": c.GLOBAL_SEED},
        weighting_desc="none (blend weights apply to whole layouts, not individual edges)",
        fallback_notes="none needed -- inherits each input layout's own disconnected-graph handling",
    )
    return coords, meta
