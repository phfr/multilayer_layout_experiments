"""Layouts driven by the real pathways_a biological annotation (Reactome-
style pathway names), not structural proxies. pathways_a only annotates
~428/900 nodes directly -- notably ZERO metabolites, since it's a
gene-centric annotation -- so both layouts here have an explicit strategy
for the coverage gap rather than silently ignoring three-quarters of the
graph.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import umap
from sklearn.decomposition import TruncatedSVD

import common as c

# fa2pathway_1: real-edge attraction boost for shared pathways.
PATHWAY_BOOST = 3.0

# pathway_landscape_1: TF-IDF + SVD dimensionality before the final UMAP step.
SVD_COMPONENTS = 50


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def run_fa2pathway_1(G, node_layer, nodes_path):
    """ForceAtlas2 with real interaction edges' attraction boosted by the
    Jaccard similarity of their endpoints' pathways_a sets, on top of the
    unweighted baseline. Real topology stays primary -- this only reinforces
    it, never replaces it, so unannotated nodes/edges behave exactly like
    fa2_1.
    """
    iso = c.isolated_nodes(G)
    Gc = G.subgraph([n for n in G.nodes() if n not in iso]).copy()

    n_boosted = 0
    for u, v, data in Gc.edges(data=True):
        j = _jaccard(
            Gc.nodes[u].get("pathways", frozenset()), Gc.nodes[v].get("pathways", frozenset())
        )
        if j > 0:
            n_boosted += 1
        data["pathway_weight"] = 1.0 + PATHWAY_BOOST * j

    xy = nx.forceatlas2_layout(Gc, max_iter=100, seed=c.GLOBAL_SEED, dim=2, weight="pathway_weight")
    coords, z_by_layer = c.layer_z_stack(xy, node_layer)
    coords.update(c.place_isolated_ring_per_layer(xy, node_layer, iso, z_by_layer))

    meta = dict(
        method=(
            "ForceAtlas2 with pathway-augmented edge weights: real interaction "
            "edges get attraction boosted by Jaccard similarity of endpoints' "
            "pathways_a annotation (real topology stays primary), z-stacked"
        ),
        library_call="nx.forceatlas2_layout(G, max_iter=100, seed=42, dim=2, weight='pathway_weight')",
        params={"max_iter": 100, "seed": c.GLOBAL_SEED, "dim": 2, "pathway_boost": PATHWAY_BOOST},
        weighting_desc=f"attraction weight = 1.0 + {PATHWAY_BOOST} * Jaccard(endpoint pathway sets)",
        fallback_notes=(
            f"{n_boosted}/{Gc.number_of_edges()} real edges got a nonzero pathway-"
            f"similarity boost; {len(iso)} isolated nodes ring-placed per layer3 at "
            f"z={z_by_layer}"
        ),
    )
    return coords, meta


def _build_pathway_matrix(G):
    nodes = sorted(G.nodes(), key=lambda n: int(n))
    node_idx = {nid: i for i, nid in enumerate(nodes)}
    vocab = sorted({t for n in nodes for t in G.nodes[n].get("pathways", frozenset())})
    term_idx = {t: i for i, t in enumerate(vocab)}
    n, d = len(nodes), len(vocab)

    X = np.zeros((n, d))
    has_direct = np.zeros(n, dtype=bool)
    for nid in nodes:
        pw = G.nodes[nid].get("pathways", frozenset())
        if pw:
            i = node_idx[nid]
            for t in pw:
                X[i, term_idx[t]] = 1.0
            has_direct[i] = True

    # Propagate a pathway profile to unannotated nodes via 1-hop, then 2-hop
    # neighbor mean-pooling (so e.g. a metabolite infers a profile from the
    # proteins it actually interacts with).
    has_signal = has_direct.copy()
    for _hop in range(2):
        still_empty = np.where(~has_signal)[0]
        if len(still_empty) == 0:
            break
        updates = {}
        for i in still_empty:
            nid = nodes[i]
            neighbor_rows = [
                X[node_idx[nb]] for nb in G.neighbors(nid) if has_signal[node_idx[nb]]
            ]
            if neighbor_rows:
                updates[i] = np.mean(neighbor_rows, axis=0)
        for i, row in updates.items():
            X[i] = row
            has_signal[i] = True

    return nodes, X, has_signal, has_direct, vocab


def run_pathway_landscape_1(G, node_layer, nodes_path):
    """Pure 'functional' landscape from real pathways_a annotation (not
    structural proxies like landscape_1/rwr_1): TF-IDF-weighted pathway
    multi-hot -> TruncatedSVD -> UMAP(3D). Similarity here means real
    biological pathway co-membership, propagated through the graph to cover
    the ~472 nodes (incl. all 300 metabolites) with no direct annotation.
    """
    nodes, X, has_signal, has_direct, vocab = _build_pathway_matrix(G)

    n_annotated = int(has_direct.sum())
    df = (X[has_direct] > 0).sum(axis=0) if n_annotated else np.zeros(len(vocab))
    idf = np.log((1 + n_annotated) / (1 + df)) + 1.0
    X_weighted = X * idf

    informative_idx = np.where(has_signal)[0]
    informative_nodes = [nodes[i] for i in informative_idx]
    X_informative = X_weighted[informative_idx]

    n_components = max(2, min(SVD_COMPONENTS, X_informative.shape[0] - 1, len(vocab) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=c.GLOBAL_SEED)
    reduced = svd.fit_transform(X_informative)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=c.N2V_UMAP_NEIGHBORS,
        min_dist=c.N2V_UMAP_MIN_DIST,
        spread=c.N2V_UMAP_SPREAD,
        metric="cosine",
        random_state=c.GLOBAL_SEED,
        verbose=False,
    )
    emb = reducer.fit_transform(reduced)
    informative_pos = {n: emb[i] for i, n in enumerate(informative_nodes)}

    remaining = [n for n in G.nodes() if n not in informative_pos]
    fallback_pos = c.deterministic_fallback_shell(informative_pos, remaining, G)

    coords = dict(informative_pos)
    coords.update(fallback_pos)

    n_propagated = int(has_signal.sum()) - n_annotated
    meta = dict(
        method=(
            "Pathway functional landscape: TF-IDF-weighted pathways_a multi-hot "
            "(propagated to unannotated nodes via 1-2 hop neighbor mean-pooling) "
            "-> TruncatedSVD -> UMAP(3D) -- similarity in real biological pathway "
            "space, not structural proxies"
        ),
        library_call=(
            f"TruncatedSVD(n_components={n_components}) -> "
            "umap.UMAP(n_components=3, metric='cosine', random_state=42)"
        ),
        params={
            "svd_components": n_components,
            "vocab_size": len(vocab),
            "n_neighbors": c.N2V_UMAP_NEIGHBORS,
            "min_dist": c.N2V_UMAP_MIN_DIST,
            "spread": c.N2V_UMAP_SPREAD,
        },
        weighting_desc="none (feature-space embedding)",
        fallback_notes=(
            f"{n_annotated} nodes have direct pathways_a annotation; {n_propagated} "
            f"more got an inferred profile via 1-2 hop neighbor mean-pooling; "
            f"{len(remaining)} nodes with no annotated node within 2 hops placed via "
            f"Fibonacci-sphere shell"
        ),
    )
    return coords, meta
