"""Layered node2vec layout: node2vec+UMAP is run SEPARATELY on the first
and last layer's same-layer subnetworks (per --layer-order; each embedded in
full 3D), each result's z-axis is squeezed into a thin slab, the two slabs
are offset apart along a global z-axis (first layer negative, last layer
positive -- matching the sign convention used by fa2_1/fa2_2/fa2_3), and
every node of a MIDDLE layer is placed in between via a Tutte/harmonic-style
iterative relaxation: its position converges to the average position of its
graph neighbors, with the propagated signal from the two fixed layers. This
is the classic Tutte embedding idea (interior points = average of neighbors,
given a fixed boundary) applied to the middle layer(s) specifically, so
their placement is driven by real connectivity rather than an arbitrary
fallback. With the default transcript/protein/metabolite order the
transcript and metabolite layers are the fixed slabs and proteins relax
between them.

Disconnected handling: each slab layer's subnetwork uses Pattern B on its
own (giant component embedded, the rest on its own fallback shell); a slab
layer whose giant component is too small for UMAP falls back to a 3D spring
layout; middle-layer components with no path to either slab go on a
Fibonacci-sphere shell around the combined anchor centroid.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import umap

import common as c
from method_node2vec import _node2vec_embedding

SLAB_THICKNESS_FRACTION = 0.08
Z_LAYER_OFFSET_FRACTION = 0.6
MIDDLE_RELAX_ITERS = 150
MIDDLE_RELAX_TOL = 1e-4
# Below this many giant-component nodes node2vec+UMAP is meaningless (UMAP
# needs n_neighbors < n and a handful of points give a degenerate manifold):
# use a plain 3D spring layout for that slab instead.
MIN_NODES_FOR_UMAP = 10


def _embed_layer_3d(H: nx.Graph, *, p: float, q: float):
    """node2vec+UMAP(3D) on H's own giant component; H's own remaining
    nodes (isolated/small components within H) placed via H's own
    deterministic fallback shell. Returns (pos, n_fallback, note)."""
    if H.number_of_nodes() == 0:
        return {}, 0, "empty layer"
    giant = c.giant_component_nodes(H)
    if len(giant) < MIN_NODES_FOR_UMAP:
        giant_nodes = sorted(giant, key=lambda n: int(n))
        raw = nx.spring_layout(H.subgraph(giant_nodes), dim=3, seed=c.GLOBAL_SEED)
        giant_pos = {n: np.asarray(raw[n], dtype=float) for n in giant_nodes}
        note = f"giant component of {len(giant)} nodes < {MIN_NODES_FOR_UMAP}: spring_layout(dim=3) instead of node2vec+UMAP"
    else:
        giant, giant_nodes, vectors = _node2vec_embedding(H, p=p, q=q)
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(c.N2V_UMAP_NEIGHBORS, len(giant_nodes) - 1),
            min_dist=c.N2V_UMAP_MIN_DIST,
            spread=c.N2V_UMAP_SPREAD,
            metric="cosine",
            random_state=c.GLOBAL_SEED,
            verbose=False,
        )
        emb = reducer.fit_transform(vectors)
        giant_pos = {n: emb[i] for i, n in enumerate(giant_nodes)}
        note = "node2vec+UMAP"
    remaining = [n for n in H.nodes() if n not in giant]
    fallback_pos = c.deterministic_fallback_shell(giant_pos, remaining, H)
    pos = dict(giant_pos)
    pos.update(fallback_pos)
    return pos, len(remaining), note


def _squeeze_and_offset(pos: dict[str, np.ndarray], *, z_sign: float, xy_spread: float) -> dict[str, np.ndarray]:
    arr = np.array(list(pos.values()))
    z_vals = arr[:, 2]
    z_center = float(z_vals.mean())
    z_range = float(z_vals.max() - z_vals.min()) or 1.0
    slab_half_thickness = 0.5 * SLAB_THICKNESS_FRACTION * xy_spread
    offset = z_sign * Z_LAYER_OFFSET_FRACTION * xy_spread
    result = {}
    for n, p in pos.items():
        x, y, z = p
        z_norm = (z - z_center) / z_range  # roughly in [-0.5, 0.5]
        result[n] = np.array([x, y, z_norm * 2 * slab_half_thickness + offset])
    return result


def run_n2vlayered_1(G, node_layer, nodes_path):
    order = c.layer_order(G)
    if len(order) < 2:
        raise ValueError("n2vlayered_1 needs at least two layers (a bottom and a top slab)")
    bottom, top = order[0], order[-1]
    middle_layers = order[1:-1]
    bottom_nodes = [n for n, l in node_layer.items() if l == bottom]
    top_nodes = [n for n, l in node_layer.items() if l == top]
    middle_nodes = [n for n, l in node_layer.items() if l in middle_layers]

    bottom_pos_raw, bottom_fallback_n, bottom_note = _embed_layer_3d(G.subgraph(bottom_nodes), p=1, q=1)
    top_pos_raw, top_fallback_n, top_note = _embed_layer_3d(G.subgraph(top_nodes), p=1, q=1)

    bottom_arr = np.array(list(bottom_pos_raw.values()))
    top_arr = np.array(list(top_pos_raw.values()))
    xy_spread = max(
        float(np.ptp(bottom_arr[:, :2], axis=0).max()),
        float(np.ptp(top_arr[:, :2], axis=0).max()),
    ) or 1.0

    bottom_pos = _squeeze_and_offset(bottom_pos_raw, z_sign=-1.0, xy_spread=xy_spread)
    top_pos = _squeeze_and_offset(top_pos_raw, z_sign=1.0, xy_spread=xy_spread)

    anchors: dict[str, np.ndarray] = {**bottom_pos, **top_pos}
    anchor_centroid = np.array(list(anchors.values())).mean(axis=0)

    # --- middle-layer placement: Tutte/harmonic relaxation, boundary = the two fixed slabs ---
    middle_sub = G.subgraph(middle_nodes)
    middle_components = list(nx.connected_components(middle_sub)) if middle_nodes else []

    reachable_nodes: set = set()
    unreachable_nodes: set = set()
    for comp in middle_components:
        touches_anchor = any(nb in anchors for m in comp for nb in G.neighbors(m))
        (reachable_nodes if touches_anchor else unreachable_nodes).update(comp)

    middle_pos: dict[str, np.ndarray] = {}
    for m in reachable_nodes:
        direct_anchor_neighbors = [anchors[nb] for nb in G.neighbors(m) if nb in anchors]
        middle_pos[m] = (
            np.mean(direct_anchor_neighbors, axis=0) if direct_anchor_neighbors else anchor_centroid.copy()
        )

    for _ in range(MIDDLE_RELAX_ITERS):
        max_delta = 0.0
        for m in sorted(reachable_nodes, key=lambda n: int(n)):
            neighbor_positions = [
                anchors[nb] if nb in anchors else middle_pos[nb]
                for nb in G.neighbors(m)
                if nb in anchors or nb in middle_pos
            ]
            if not neighbor_positions:
                continue
            new_pos = np.mean(neighbor_positions, axis=0)
            max_delta = max(max_delta, float(np.linalg.norm(new_pos - middle_pos[m])))
            middle_pos[m] = new_pos
        if max_delta < MIDDLE_RELAX_TOL:
            break

    fallback_pos = {}
    if unreachable_nodes:
        fallback_pos = c.deterministic_fallback_shell(anchors, unreachable_nodes, G)

    coords: dict[str, np.ndarray] = {}
    coords.update(bottom_pos)
    coords.update(top_pos)
    coords.update(middle_pos)
    coords.update(fallback_pos)

    middle_desc = "/".join(middle_layers) if middle_layers else "(none)"
    meta = dict(
        method=(
            f"Layered node2vec+UMAP: separate 3D embeddings for the {bottom}-only and "
            f"{top}-only same-layer subnetworks, each z-squeezed into a slab and "
            f"offset apart ({bottom} below, {top} above); {middle_desc} nodes placed via "
            "Tutte/harmonic-style iterative relaxation (average of graph neighbors) "
            "from the two fixed slabs"
        ),
        library_call=(
            f"Node2Vec({bottom}_subgraph).fit() -> umap.UMAP(3D); same for {top}_subgraph; "
            "middle_pos[m] = mean(neighbor positions), Gauss-Seidel iterated to convergence"
        ),
        params={
            "p": 1,
            "q": 1,
            "slab_layers": (bottom, top),
            "middle_layers": middle_layers,
            "slab_thickness_fraction": SLAB_THICKNESS_FRACTION,
            "z_layer_offset_fraction": Z_LAYER_OFFSET_FRACTION,
            "middle_relax_iters": MIDDLE_RELAX_ITERS,
            "middle_relax_tol": MIDDLE_RELAX_TOL,
            "min_nodes_for_umap": MIN_NODES_FOR_UMAP,
            "seed": c.GLOBAL_SEED,
        },
        weighting_desc="none (unweighted random walks per subnetwork; unweighted neighbor averaging for middle layers)",
        fallback_notes=(
            f"{bottom} subnetwork ({bottom_note}): {bottom_fallback_n} nodes outside its own giant "
            f"component fallback-shell placed; {top} subnetwork ({top_note}): "
            f"{top_fallback_n} similarly; {len(reachable_nodes)} middle-layer nodes "
            f"placed via harmonic relaxation (path exists to a {bottom}/{top} node), "
            f"{len(unreachable_nodes)} middle-layer nodes with no path to either slab "
            f"placed via Fibonacci-sphere shell around the combined anchor centroid"
        ),
    )
    return coords, meta
