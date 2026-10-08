"""VRNetzer-inspired "functional landscape" layouts: embed nodes by
ANNOTATION/FEATURE similarity rather than graph-distance/topology. Two
nodes with similar profiles land close together even if they're in
different connected components or never interact directly -- a
fundamentally different signal from every distance-based layout in this
package (Pfeil et al., "Visualizing biological data in a virtual
environment", Nature Communications 2021, uses this idea with GO-annotation
feature vectors and random-walk-with-restart feature vectors; we substitute
edge type / layer / node type / community / degree as our available proxy
signal, since we have no external annotation database). The edge-type and
node-type blocks are data-driven and simply absent when the input has no
such column (see common.ColumnSpec).
"""

from __future__ import annotations

import community as community_louvain
import networkx as nx
import numpy as np
import umap

import common as c


def build_feature_matrix(G: nx.Graph):
    """Per-node feature vector: fraction of incident edges of each edge-type
    value, layer one-hot, node-type one-hot (data-driven vocabularies; no
    edge-type / type column -> no such block), Louvain community one-hot,
    log-degree. Every node (including isolated ones) gets a valid vector."""
    nodes = sorted(G.nodes(), key=c.node_key)
    partition = community_louvain.best_partition(G, random_state=c.GLOBAL_SEED)
    community_ids = sorted(set(partition.values()))
    community_idx = {cid: i for i, cid in enumerate(community_ids)}
    # Layer one-hot first, then raw node-type one-hot (a finer split within a
    # layer, e.g. protein_bridge vs protein_differentially_expressed).
    type_values = list(c.layer_order(G)) + sorted(
        {G.nodes[n].get("type", "") for n in nodes} - {""}
    )
    type_idx = {t: i for i, t in enumerate(type_values)}

    edge_type_idx: dict[str, int] = {}
    per_node_counts = []
    for n in nodes:
        counts: dict[str, int] = {}
        for _, _, data in G.edges(n, data=True):
            et = data.get("edge_type", "")
            if not et:  # no edge-type column (or untyped edge): no composition signal
                continue
            counts[et] = counts.get(et, 0) + 1
            if et not in edge_type_idx:
                edge_type_idx[et] = len(edge_type_idx)
        per_node_counts.append(counts)

    dim = len(edge_type_idx) + len(type_values) + len(community_ids) + 1
    X = np.zeros((len(nodes), dim))
    n_et = len(edge_type_idx)
    n_ta = len(type_values)

    for row_i, (n, counts) in enumerate(zip(nodes, per_node_counts)):
        deg = G.degree(n)
        for et, cnt in counts.items():
            X[row_i, edge_type_idx[et]] = cnt / deg if deg else 0.0
        X[row_i, n_et + type_idx[G.nodes[n]["layer"]]] = 1.0
        ta = G.nodes[n].get("type", "")
        if ta in type_idx:
            X[row_i, n_et + type_idx[ta]] = 1.0
        X[row_i, n_et + n_ta + community_idx[partition[n]]] = 1.0
        X[row_i, -1] = np.log1p(deg)

    return nodes, X, {"edge_types": n_et, "layer+type": n_ta, "communities": len(community_ids)}


def run_landscape_1(G, node_layer, nodes_path):
    nodes, X, dims = build_feature_matrix(G)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=c.N2V_UMAP_NEIGHBORS,
        min_dist=c.N2V_UMAP_MIN_DIST,
        spread=c.N2V_UMAP_SPREAD,
        metric="cosine",
        random_state=c.GLOBAL_SEED,
        verbose=False,
    )
    emb = reducer.fit_transform(X)
    coords = {n: emb[i] for i, n in enumerate(nodes)}

    meta = dict(
        method=(
            "VRNetzer-style functional landscape: per-node feature vector "
            "(edge-type composition fractions + layer/type one-hot + Louvain "
            "community one-hot + log-degree) embedded via UMAP -- similarity "
            "in ANNOTATION space, not graph-distance space"
        ),
        library_call="umap.UMAP(n_components=3, metric='cosine', random_state=42) on a hand-built feature matrix",
        params={
            "feature_dims": int(X.shape[1]),
            "feature_breakdown": dims,
            "n_neighbors": c.N2V_UMAP_NEIGHBORS,
            "min_dist": c.N2V_UMAP_MIN_DIST,
            "spread": c.N2V_UMAP_SPREAD,
        },
        weighting_desc="none (feature-space embedding, no graph edge weighting)",
        fallback_notes="none needed -- every node (incl. isolated) has a valid feature vector",
    )
    return coords, meta


def build_rwr_feature_matrix(G: nx.Graph, *, restart_prob: float = 0.9):
    """Personalized-PageRank (random-walk-with-restart) feature vector per
    node: row i = stationary visitation distribution of a walk restarting
    at node i with probability `restart_prob`. nx.pagerank's `alpha` is the
    probability of CONTINUING the walk, so alpha = 1 - restart_prob."""
    nodes = sorted(G.nodes(), key=c.node_key)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    X = np.zeros((n, n))
    alpha = 1.0 - restart_prob
    for i, source in enumerate(nodes):
        pr = nx.pagerank(G, alpha=alpha, personalization={source: 1.0})
        for target, p in pr.items():
            X[i, idx[target]] = p
    return nodes, X


def run_rwr_1(G, node_layer, nodes_path):
    restart_prob = 0.9
    nodes, X = build_rwr_feature_matrix(G, restart_prob=restart_prob)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=c.N2V_UMAP_NEIGHBORS,
        min_dist=c.N2V_UMAP_MIN_DIST,
        spread=c.N2V_UMAP_SPREAD,
        metric="cosine",
        random_state=c.GLOBAL_SEED,
        verbose=False,
    )
    emb = reducer.fit_transform(X)
    coords = {n: emb[i] for i, n in enumerate(nodes)}

    meta = dict(
        method=(
            "Random-walk-with-restart feature landscape: personalized-PageRank "
            "visitation distribution per node embedded via UMAP -- a diffusion-based "
            "feature space rather than raw topology or a fixed annotation vector"
        ),
        library_call=f"nx.pagerank(G, alpha={1-restart_prob}, personalization={{source:1.0}}) per node -> umap.UMAP(3D)",
        params={
            "restart_prob": restart_prob,
            "n_neighbors": c.N2V_UMAP_NEIGHBORS,
            "min_dist": c.N2V_UMAP_MIN_DIST,
            "spread": c.N2V_UMAP_SPREAD,
        },
        weighting_desc="none (feature-space embedding)",
        fallback_notes="none needed -- personalized PageRank is well-defined for isolated/dangling nodes",
    )
    return coords, meta
