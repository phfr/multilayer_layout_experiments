"""node2vec embedding + dimensionality reduction (UMAP, PHATE, PaCMAP,
DensMAP, or openTSNE) to 3D. Runs on the giant component only (random walks
can't leave a disconnected component, so embeddings for isolated/small-
component nodes would be degenerate); every other node is placed via
common.deterministic_fallback_shell, same as kk3d/mds/spectral.
"""

from __future__ import annotations

import numpy as np
import pacmap
import phate
import umap
from node2vec import Node2Vec
from openTSNE import TSNE

import common as c


def _node2vec_embedding(G, *, p: float, q: float):
    giant = c.giant_component_nodes(G)
    giant_nodes = sorted(giant, key=c.node_key)
    sub = c.ordered_subgraph(G, giant_nodes)

    n2v = Node2Vec(
        sub,
        dimensions=c.N2V_DIMENSIONS,
        walk_length=c.N2V_WALK_LENGTH,
        num_walks=c.N2V_NUM_WALKS,
        p=p,
        q=q,
        seed=c.GLOBAL_SEED,
        quiet=True,  # raw (non-tqdm-aware) node2vec/gensim output buries the persistent progress bar
        workers=1,
    )
    model = n2v.fit(window=10, min_count=1, seed=c.GLOBAL_SEED, workers=1)
    vectors = np.array([model.wv[str(n)] for n in giant_nodes])
    return giant, giant_nodes, vectors


def _node2vec_umap(G, node_layer, *, p: float, q: float):
    giant, giant_nodes, vectors = _node2vec_embedding(G, p=p, q=q)

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

    remaining = [n for n in G.nodes() if n not in giant]
    fallback_pos = c.deterministic_fallback_shell(giant_pos, remaining, G)

    coords = dict(giant_pos)
    coords.update(fallback_pos)

    meta = dict(
        method=f"node2vec (p={p}, q={q}) embedding + UMAP(3D), giant component only",
        library_call=(
            "Node2Vec(giant_subgraph, dimensions=64, walk_length=30, num_walks=200, "
            f"p={p}, q={q}, seed=42, workers=1).fit(window=10, min_count=1, seed=42, "
            f"workers=1) -> umap.UMAP(n_components=3, n_neighbors={c.N2V_UMAP_NEIGHBORS}, "
            f"min_dist={c.N2V_UMAP_MIN_DIST}, spread={c.N2V_UMAP_SPREAD}, "
            "metric='cosine', random_state=42)"
        ),
        params={
            "p": p,
            "q": q,
            "dimensions": c.N2V_DIMENSIONS,
            "walk_length": c.N2V_WALK_LENGTH,
            "num_walks": c.N2V_NUM_WALKS,
            "seed": c.GLOBAL_SEED,
            "workers": 1,
            "umap_n_neighbors": c.N2V_UMAP_NEIGHBORS,
            "umap_min_dist": c.N2V_UMAP_MIN_DIST,
            "umap_spread": c.N2V_UMAP_SPREAD,
            "umap_metric": "cosine",
        },
        weighting_desc="none (unweighted random walks)",
        fallback_notes=(
            f"{len(remaining)} nodes outside the giant component (random walks can't "
            f"leave a disconnected component) placed via Fibonacci-sphere shell, "
            f"radius_factor={c.SHELL_RADIUS_FACTOR}"
        ),
    )
    return coords, meta


def run_n2vbal_1(G, node_layer, nodes_path):
    return _node2vec_umap(G, node_layer, p=1, q=1)


def run_n2vbfs_1(G, node_layer, nodes_path):
    return _node2vec_umap(G, node_layer, p=0.25, q=4)


def run_n2vdfs_1(G, node_layer, nodes_path):
    return _node2vec_umap(G, node_layer, p=4, q=0.25)


def _node2vec_densmap(G, node_layer, *, p: float, q: float):
    """DensMAP (Narayan, Berger & Cho, Nat. Biotechnol. 2021): a
    density-preserving UMAP variant -- adds a density-preservation term to
    UMAP's loss so regions of varying local point-density in the embedding
    reflect true local density in the original embedding space, rather than
    UMAP's default behavior of visually equalizing density everywhere.
    Already available in the installed umap-learn via `densmap=True`.
    """
    giant, giant_nodes, vectors = _node2vec_embedding(G, p=p, q=q)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=c.N2V_UMAP_NEIGHBORS,
        min_dist=c.N2V_UMAP_MIN_DIST,
        spread=c.N2V_UMAP_SPREAD,
        metric="cosine",
        random_state=c.GLOBAL_SEED,
        densmap=True,
        verbose=False,
    )
    emb = reducer.fit_transform(vectors)
    giant_pos = {n: emb[i] for i, n in enumerate(giant_nodes)}

    remaining = [n for n in G.nodes() if n not in giant]
    fallback_pos = c.deterministic_fallback_shell(giant_pos, remaining, G)

    coords = dict(giant_pos)
    coords.update(fallback_pos)

    meta = dict(
        method=f"node2vec (p={p}, q={q}) embedding + DensMAP(3D), giant component only",
        library_call=(
            "Node2Vec(giant_subgraph, ...).fit(...) -> umap.UMAP(n_components=3, "
            f"n_neighbors={c.N2V_UMAP_NEIGHBORS}, min_dist={c.N2V_UMAP_MIN_DIST}, "
            f"spread={c.N2V_UMAP_SPREAD}, densmap=True, metric='cosine', random_state=42)"
        ),
        params={
            "p": p,
            "q": q,
            "dimensions": c.N2V_DIMENSIONS,
            "walk_length": c.N2V_WALK_LENGTH,
            "num_walks": c.N2V_NUM_WALKS,
            "seed": c.GLOBAL_SEED,
            "workers": 1,
            "umap_n_neighbors": c.N2V_UMAP_NEIGHBORS,
            "umap_min_dist": c.N2V_UMAP_MIN_DIST,
            "umap_spread": c.N2V_UMAP_SPREAD,
            "umap_densmap": True,
        },
        weighting_desc="none (unweighted random walks)",
        fallback_notes=(
            f"{len(remaining)} nodes outside the giant component placed via "
            f"Fibonacci-sphere shell, radius_factor={c.SHELL_RADIUS_FACTOR}"
        ),
    )
    return coords, meta


def run_n2vbal_densmap_1(G, node_layer, nodes_path):
    return _node2vec_densmap(G, node_layer, p=1, q=1)


def _node2vec_phate(G, node_layer, *, p: float, q: float):
    """PHATE (Moon et al., Nat. Biotechnol. 2019) instead of UMAP as the
    reducer: a diffusion-operator-based method purpose-built for biological
    data, explicitly designed to preserve both local AND global/trajectory
    structure (UMAP/t-SNE are local-neighborhood-focused by comparison).
    """
    giant, giant_nodes, vectors = _node2vec_embedding(G, p=p, q=q)

    reducer = phate.PHATE(n_components=3, random_state=c.GLOBAL_SEED, verbose=False)
    emb = reducer.fit_transform(vectors)
    giant_pos = {n: emb[i] for i, n in enumerate(giant_nodes)}

    remaining = [n for n in G.nodes() if n not in giant]
    fallback_pos = c.deterministic_fallback_shell(giant_pos, remaining, G)

    coords = dict(giant_pos)
    coords.update(fallback_pos)

    meta = dict(
        method=f"node2vec (p={p}, q={q}) embedding + PHATE(3D), giant component only",
        library_call=(
            "Node2Vec(giant_subgraph, dimensions=64, walk_length=30, num_walks=200, "
            f"p={p}, q={q}, seed=42, workers=1).fit(window=10, min_count=1, seed=42, "
            "workers=1) -> phate.PHATE(n_components=3, random_state=42)"
        ),
        params={
            "p": p,
            "q": q,
            "dimensions": c.N2V_DIMENSIONS,
            "walk_length": c.N2V_WALK_LENGTH,
            "num_walks": c.N2V_NUM_WALKS,
            "seed": c.GLOBAL_SEED,
            "workers": 1,
        },
        weighting_desc="none (unweighted random walks)",
        fallback_notes=(
            f"{len(remaining)} nodes outside the giant component (random walks can't "
            f"leave a disconnected component) placed via Fibonacci-sphere shell, "
            f"radius_factor={c.SHELL_RADIUS_FACTOR}"
        ),
    )
    return coords, meta


def run_n2vbal_phate_1(G, node_layer, nodes_path):
    return _node2vec_phate(G, node_layer, p=1, q=1)


def _node2vec_pacmap(G, node_layer, *, p: float, q: float):
    """PaCMAP (Wang et al., JMLR 2021) instead of UMAP as the reducer: the
    same technique the frontend already uses client-side for its own
    in-browser layout calc (src/windows/LayoutCalcWindow.ts). PaCMAP
    explicitly balances local AND mid/global structure via its
    near/mid-near/further point-pair sampling, which often produces more
    evenly-spread embeddings than UMAP (less prone to the "tight clusters
    with big gaps" clumping UMAP can produce -- see evaluate_umap_params.py).
    """
    giant, giant_nodes, vectors = _node2vec_embedding(G, p=p, q=q)

    reducer = pacmap.PaCMAP(
        n_components=3,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        distance="angular",  # cosine-like, consistent with the UMAP variants
        random_state=c.GLOBAL_SEED,
    )
    emb = reducer.fit_transform(vectors)
    giant_pos = {n: emb[i] for i, n in enumerate(giant_nodes)}

    remaining = [n for n in G.nodes() if n not in giant]
    fallback_pos = c.deterministic_fallback_shell(giant_pos, remaining, G)

    coords = dict(giant_pos)
    coords.update(fallback_pos)

    meta = dict(
        method=f"node2vec (p={p}, q={q}) embedding + PaCMAP(3D), giant component only",
        library_call=(
            "Node2Vec(giant_subgraph, dimensions=64, walk_length=30, num_walks=200, "
            f"p={p}, q={q}, seed=42, workers=1).fit(window=10, min_count=1, seed=42, "
            "workers=1) -> pacmap.PaCMAP(n_components=3, n_neighbors=10, "
            "distance='angular', random_state=42)"
        ),
        params={
            "p": p,
            "q": q,
            "dimensions": c.N2V_DIMENSIONS,
            "walk_length": c.N2V_WALK_LENGTH,
            "num_walks": c.N2V_NUM_WALKS,
            "seed": c.GLOBAL_SEED,
            "workers": 1,
            "pacmap_n_neighbors": 10,
            "pacmap_MN_ratio": 0.5,
            "pacmap_FP_ratio": 2.0,
            "pacmap_distance": "angular",
        },
        weighting_desc="none (unweighted random walks)",
        fallback_notes=(
            f"{len(remaining)} nodes outside the giant component (random walks can't "
            f"leave a disconnected component) placed via Fibonacci-sphere shell, "
            f"radius_factor={c.SHELL_RADIUS_FACTOR}"
        ),
    )
    return coords, meta


def run_n2vbal_pacmap_1(G, node_layer, nodes_path):
    return _node2vec_pacmap(G, node_layer, p=1, q=1)


def run_n2vbfs_pacmap_1(G, node_layer, nodes_path):
    return _node2vec_pacmap(G, node_layer, p=0.25, q=4)


def run_n2vdfs_pacmap_1(G, node_layer, nodes_path):
    return _node2vec_pacmap(G, node_layer, p=4, q=0.25)


def _node2vec_tsne(G, node_layer, *, p: float, q: float):
    """Classic t-SNE (van der Maaten & Hinton 2008) via openTSNE (Poličar,
    Stražar & Zupan, J. Stat. Software 2024) -- actively maintained, much
    faster than sklearn's TSNE. Known for producing tighter, more visually
    separated clusters than UMAP/PaCMAP tend to give here, at some cost to
    inter-cluster distance fidelity. `negative_gradient_method='bh'`
    (Barnes-Hut, tree-based) is used explicitly since openTSNE's docs note
    the FFT/interpolation acceleration is a 2D-only grid trick that isn't
    reliable above 2D -- Barnes-Hut's spatial tree generalizes fine to 3D
    and is openTSNE's own "auto" choice at this sample size anyway (n <
    10,000).
    """
    giant, giant_nodes, vectors = _node2vec_embedding(G, p=p, q=q)

    tsne = TSNE(
        n_components=3,
        perplexity=30,
        metric="cosine",
        negative_gradient_method="bh",
        initialization="pca",
        random_state=c.GLOBAL_SEED,
        n_jobs=1,
        verbose=False,
    )
    emb = np.asarray(tsne.fit(vectors))
    giant_pos = {n: emb[i] for i, n in enumerate(giant_nodes)}

    remaining = [n for n in G.nodes() if n not in giant]
    fallback_pos = c.deterministic_fallback_shell(giant_pos, remaining, G)

    coords = dict(giant_pos)
    coords.update(fallback_pos)

    meta = dict(
        method=f"node2vec (p={p}, q={q}) embedding + openTSNE(3D), giant component only",
        library_call=(
            "Node2Vec(giant_subgraph, ...).fit(...) -> openTSNE.TSNE(n_components=3, "
            "perplexity=30, metric='cosine', negative_gradient_method='bh', "
            "initialization='pca', random_state=42)"
        ),
        params={
            "p": p,
            "q": q,
            "dimensions": c.N2V_DIMENSIONS,
            "walk_length": c.N2V_WALK_LENGTH,
            "num_walks": c.N2V_NUM_WALKS,
            "seed": c.GLOBAL_SEED,
            "tsne_perplexity": 30,
            "tsne_negative_gradient_method": "bh",
        },
        weighting_desc="none (unweighted random walks)",
        fallback_notes=(
            f"{len(remaining)} nodes outside the giant component placed via "
            f"Fibonacci-sphere shell, radius_factor={c.SHELL_RADIUS_FACTOR}"
        ),
    )
    return coords, meta


def run_n2vbal_tsne_1(G, node_layer, nodes_path):
    return _node2vec_tsne(G, node_layer, p=1, q=1)
