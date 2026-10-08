"""Precomputed-matrix embeddings: classical MDS on a shortest-path distance
matrix, and spectral embedding (Laplacian eigenmaps) on a (weighted)
adjacency matrix. Both run on the giant component only, with every other
node (isolated singles + small components) placed via
common.deterministic_fallback_shell -- the same disconnected-graph strategy
as Kamada-Kawai, since both methods here depend on global distance/affinity
structure that isn't meaningful across disconnected components.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.manifold import MDS, Isomap

import common as c


def _hop_distance_matrix(sub: nx.Graph, nodes: list[str], *, weight_attr: str | None) -> np.ndarray:
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    dist_dict = dict(nx.shortest_path_length(sub, weight=weight_attr))
    mat = np.zeros((n, n))
    for u, targets in dist_dict.items():
        i = idx[u]
        for v, d in targets.items():
            mat[i, idx[v]] = d
    return mat


def _mds(G, node_layer, *, weighted: bool):
    giant = c.giant_component_nodes(G)
    giant_nodes = sorted(giant, key=c.node_key)
    sub = c.ordered_subgraph(G, giant_nodes)

    weight_attr = None
    weighting_desc = "unweighted hop count"
    if weighted:
        c.set_distance_weight(sub, node_layer, attr="mds_hop_weight")
        weight_attr = "mds_hop_weight"
        weighting_desc = f"hop length: within={c.DISTANCE_WITHIN}, cross={c.DISTANCE_CROSS}"

    matrix = _hop_distance_matrix(sub, giant_nodes, weight_attr=weight_attr)

    model = MDS(
        n_components=3,
        dissimilarity="precomputed",
        random_state=c.GLOBAL_SEED,
        n_init=4,
        normalized_stress=False,
    )
    emb = model.fit_transform(matrix)
    giant_pos = {n: emb[i] for i, n in enumerate(giant_nodes)}

    remaining = [n for n in G.nodes() if n not in giant]
    fallback_pos = c.deterministic_fallback_shell(giant_pos, remaining, G)

    coords = dict(giant_pos)
    coords.update(fallback_pos)

    meta = dict(
        method="Classical MDS on shortest-path distance matrix (giant component only)",
        library_call=(
            "sklearn.manifold.MDS(n_components=3, dissimilarity='precomputed', "
            "random_state=42, n_init=4)"
        ),
        params={"n_components": 3, "random_state": c.GLOBAL_SEED, "n_init": 4, "weight": weight_attr},
        weighting_desc=weighting_desc,
        fallback_notes=(
            f"{len(remaining)} nodes outside the giant component placed via "
            f"Fibonacci-sphere shell, radius_factor={c.SHELL_RADIUS_FACTOR}"
        ),
    )
    return coords, meta


def run_mds_1(G, node_layer, nodes_path):
    return _mds(G, node_layer, weighted=False)


def run_mds_2(G, node_layer, nodes_path):
    return _mds(G, node_layer, weighted=True)


def _spectral(G, node_layer, *, weighted: bool):
    giant = c.giant_component_nodes(G)
    giant_nodes = sorted(giant, key=c.node_key)
    sub = c.ordered_subgraph(G, giant_nodes)

    giant_pos = c.spectral_positions(sub, 3, weighted=weighted, node_layer=node_layer)

    remaining = [n for n in G.nodes() if n not in giant]
    fallback_pos = c.deterministic_fallback_shell(giant_pos, remaining, G)

    coords = dict(giant_pos)
    coords.update(fallback_pos)

    weighting_desc = (
        "unweighted adjacency"
        if not weighted
        else f"adjacency weighted: within={c.ATTRACTION_WITHIN}, cross={c.ATTRACTION_CROSS}"
    )
    meta = dict(
        method="Spectral embedding (Laplacian eigenmaps) on adjacency matrix (giant component only)",
        library_call=(
            "sklearn.manifold.SpectralEmbedding(n_components=3, affinity='precomputed', "
            "random_state=42)"
        ),
        params={"n_components": 3, "random_state": c.GLOBAL_SEED},
        weighting_desc=weighting_desc,
        fallback_notes=(
            f"epsilon={c.SPECTRAL_EPSILON} background added to adjacency for numerical "
            f"stability; {len(remaining)} nodes outside the giant component placed via "
            f"Fibonacci-sphere shell, radius_factor={c.SHELL_RADIUS_FACTOR}"
        ),
    )
    return coords, meta


def run_spectral_1(G, node_layer, nodes_path):
    return _spectral(G, node_layer, weighted=False)


def run_spectral_2(G, node_layer, nodes_path):
    return _spectral(G, node_layer, weighted=True)


def run_isomap_1(G, node_layer, nodes_path):
    """Isomap (kernel PCA on a k-NN-restricted geodesic distance matrix) on
    the giant component's hop-distance matrix. A distinct embedding
    objective from both MDS (stress majorization on the FULL distance
    matrix) and spectral embedding (Laplacian eigenmaps): Isomap first
    restricts to each node's k nearest neighbors (by hop distance) before
    embedding, which can reveal different manifold structure.
    """
    giant = c.giant_component_nodes(G)
    giant_nodes = sorted(giant, key=c.node_key)
    sub = c.ordered_subgraph(G, giant_nodes)
    matrix = _hop_distance_matrix(sub, giant_nodes, weight_attr=None)

    n_neighbors = 10
    model = Isomap(n_neighbors=n_neighbors, n_components=3, metric="precomputed")
    emb = model.fit_transform(matrix)
    giant_pos = {n: emb[i] for i, n in enumerate(giant_nodes)}

    remaining = [n for n in G.nodes() if n not in giant]
    fallback_pos = c.deterministic_fallback_shell(giant_pos, remaining, G)

    coords = dict(giant_pos)
    coords.update(fallback_pos)

    meta = dict(
        method="Isomap on giant component's hop-distance matrix (k-NN-restricted geodesic embedding)",
        library_call=f"sklearn.manifold.Isomap(n_neighbors={n_neighbors}, n_components=3, metric='precomputed')",
        params={"n_neighbors": n_neighbors, "n_components": 3},
        weighting_desc="unweighted hop count",
        fallback_notes=(
            f"{len(remaining)} nodes outside the giant component placed via "
            f"Fibonacci-sphere shell, radius_factor={c.SHELL_RADIUS_FACTOR}"
        ),
    )
    return coords, meta
