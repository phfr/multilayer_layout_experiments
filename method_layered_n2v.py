"""Layered node2vec layout: node2vec+UMAP is run SEPARATELY on the
protein-only and metabolite-only same-layer subnetworks (each embedded in
full 3D), each result's z-axis is squeezed into a thin slab, the two slabs
are offset apart along a global z-axis (protein negative, metabolite
positive -- matching the sign convention used by fa2_1/fa2_2/fa2_3), and
bridge nodes are placed in between via a Tutte/harmonic-style iterative
relaxation: each bridge node's position converges to the average position of
its graph neighbors, with the propagated signal from the two fixed layers.
This is the classic Tutte embedding idea (interior points = average of
neighbors, given a fixed boundary) applied to the bridge layer specifically,
so bridge placement is driven by real connectivity rather than an arbitrary
fallback.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import umap

import common as c
from method_node2vec import _node2vec_embedding

SLAB_THICKNESS_FRACTION = 0.08
Z_LAYER_OFFSET_FRACTION = 0.6
BRIDGE_RELAX_ITERS = 150
BRIDGE_RELAX_TOL = 1e-4


def _embed_layer_3d(H: nx.Graph, *, p: float, q: float):
    """node2vec+UMAP(3D) on H's own giant component; H's own remaining
    nodes (isolated/small components within H) placed via H's own
    deterministic fallback shell."""
    giant, giant_nodes, vectors = _node2vec_embedding(H, p=p, q=q)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=c.N2V_UMAP_NEIGHBORS,
        min_dist=c.N2V_UMAP_MIN_DIST,
        spread=c.N2V_UMAP_SPREAD,
        metric="cosine",
        random_state=c.GLOBAL_SEED,
        verbose=False,
    )
    emb = reducer.fit_transform(vectors)
    giant_pos = {n: emb[i] for i, n in enumerate(giant_nodes)}
    remaining = [n for n in H.nodes() if n not in giant]
    fallback_pos = c.deterministic_fallback_shell(giant_pos, remaining, H)
    pos = dict(giant_pos)
    pos.update(fallback_pos)
    return pos, len(remaining)


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
    protein_nodes = [n for n, l in node_layer.items() if l == "protein"]
    metabolite_nodes = [n for n, l in node_layer.items() if l == "metabolite"]
    bridge_nodes = [n for n, l in node_layer.items() if l == "bridge"]

    protein_sub = G.subgraph(protein_nodes)
    metabolite_sub = G.subgraph(metabolite_nodes)

    protein_pos_raw, protein_fallback_n = _embed_layer_3d(protein_sub, p=1, q=1)
    metabolite_pos_raw, metabolite_fallback_n = _embed_layer_3d(metabolite_sub, p=1, q=1)

    protein_arr = np.array(list(protein_pos_raw.values()))
    metabolite_arr = np.array(list(metabolite_pos_raw.values()))
    xy_spread = max(
        float(protein_arr[:, :2].ptp(axis=0).max()),
        float(metabolite_arr[:, :2].ptp(axis=0).max()),
    )

    protein_pos = _squeeze_and_offset(protein_pos_raw, z_sign=-1.0, xy_spread=xy_spread)
    metabolite_pos = _squeeze_and_offset(metabolite_pos_raw, z_sign=1.0, xy_spread=xy_spread)

    anchors: dict[str, np.ndarray] = {**protein_pos, **metabolite_pos}
    anchor_centroid = np.array(list(anchors.values())).mean(axis=0)

    # --- bridge placement: Tutte/harmonic relaxation, boundary = the two fixed layers ---
    bridge_sub = G.subgraph(bridge_nodes)
    bridge_components = list(nx.connected_components(bridge_sub)) if bridge_nodes else []

    reachable_nodes: set = set()
    unreachable_nodes: set = set()
    for comp in bridge_components:
        touches_anchor = any(nb in anchors for b in comp for nb in G.neighbors(b))
        (reachable_nodes if touches_anchor else unreachable_nodes).update(comp)

    bridge_pos: dict[str, np.ndarray] = {}
    for b in reachable_nodes:
        direct_anchor_neighbors = [anchors[nb] for nb in G.neighbors(b) if nb in anchors]
        bridge_pos[b] = (
            np.mean(direct_anchor_neighbors, axis=0) if direct_anchor_neighbors else anchor_centroid.copy()
        )

    for _ in range(BRIDGE_RELAX_ITERS):
        max_delta = 0.0
        for b in sorted(reachable_nodes, key=lambda n: int(n)):
            neighbor_positions = [
                anchors[nb] if nb in anchors else bridge_pos[nb]
                for nb in G.neighbors(b)
                if nb in anchors or nb in bridge_pos
            ]
            if not neighbor_positions:
                continue
            new_pos = np.mean(neighbor_positions, axis=0)
            max_delta = max(max_delta, float(np.linalg.norm(new_pos - bridge_pos[b])))
            bridge_pos[b] = new_pos
        if max_delta < BRIDGE_RELAX_TOL:
            break

    fallback_pos = {}
    if unreachable_nodes:
        fallback_pos = c.deterministic_fallback_shell(anchors, unreachable_nodes, G)

    coords: dict[str, np.ndarray] = {}
    coords.update(protein_pos)
    coords.update(metabolite_pos)
    coords.update(bridge_pos)
    coords.update(fallback_pos)

    meta = dict(
        method=(
            "Layered node2vec+UMAP: separate 3D embeddings for the protein-only and "
            "metabolite-only same-layer subnetworks, each z-squeezed into a slab and "
            "offset apart; bridge nodes placed via Tutte/harmonic-style iterative "
            "relaxation (average of graph neighbors) from the two fixed layers"
        ),
        library_call=(
            "Node2Vec(protein_subgraph).fit() -> umap.UMAP(3D); same for metabolite_subgraph; "
            "bridge_pos[b] = mean(neighbor positions), Gauss-Seidel iterated to convergence"
        ),
        params={
            "p": 1,
            "q": 1,
            "slab_thickness_fraction": SLAB_THICKNESS_FRACTION,
            "z_layer_offset_fraction": Z_LAYER_OFFSET_FRACTION,
            "bridge_relax_iters": BRIDGE_RELAX_ITERS,
            "bridge_relax_tol": BRIDGE_RELAX_TOL,
            "seed": c.GLOBAL_SEED,
        },
        weighting_desc="none (unweighted random walks per subnetwork; unweighted neighbor averaging for bridges)",
        fallback_notes=(
            f"protein subnetwork: {protein_fallback_n} nodes outside its own giant "
            f"component fallback-shell placed; metabolite subnetwork: "
            f"{metabolite_fallback_n} similarly; {len(reachable_nodes)} bridge nodes "
            f"placed via harmonic relaxation (path exists to a protein/metabolite "
            f"node), {len(unreachable_nodes)} fully bridge-only-isolated nodes (no "
            f"path to either layer) placed via Fibonacci-sphere shell around the "
            f"combined anchor centroid"
        ),
    )
    return coords, meta
