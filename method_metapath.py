"""Metapath-biased random walks: like node2vec, but the walk's transition
probability is biased by edge_type_c (domain continuity) rather than being
purely topology-driven -- node2vec's p/q only control BFS/DFS exploration
bias, never look at edge TYPE at all. Walks here are SAME_DOMAIN_BOOST times
more likely to continue along a same-domain edge (an edge_type_c whose
two halves match, e.g. protein-protein, transcript-transcript,
metabolite-metabolite) than to cross domains, so the resulting embedding reflects domain-continuity structure
that plain node2vec can't see. The node2vec package doesn't expose
per-edge-type biasing, so this uses a small custom weighted-walk generator
feeding directly into gensim's Word2Vec (the same training call node2vec
uses internally).
"""

from __future__ import annotations

import random

import gensim
import networkx as nx
import numpy as np
import umap

import common as c

SAME_DOMAIN_BOOST = 4.0
WALK_LENGTH = 20
NUM_WALKS = 100


def _is_same_domain(edge_type_c: str) -> bool:
    if "-" not in edge_type_c:
        return False
    a, b = edge_type_c.split("-", 1)
    return a == b


def _generate_walk(G: nx.Graph, start: str, rng: random.Random) -> list[str]:
    walk = [start]
    current = start
    for _ in range(WALK_LENGTH - 1):
        neighbors = list(G.neighbors(current))
        if not neighbors:
            break
        weights = [
            SAME_DOMAIN_BOOST if _is_same_domain(G.edges[current, nb].get("edge_type_c", "")) else 1.0
            for nb in neighbors
        ]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        nxt = neighbors[-1]
        for nb, w in zip(neighbors, weights):
            acc += w
            if r <= acc:
                nxt = nb
                break
        walk.append(nxt)
        current = nxt
    return walk


def run_metapath_1(G, node_layer, nodes_path):
    giant = c.giant_component_nodes(G)
    giant_nodes = sorted(giant, key=lambda n: int(n))
    sub = G.subgraph(giant_nodes)

    rng = random.Random(c.GLOBAL_SEED)
    walks = []
    for _ in range(NUM_WALKS):
        for n in giant_nodes:
            walks.append([str(x) for x in _generate_walk(sub, n, rng)])

    model = gensim.models.Word2Vec(
        walks,
        vector_size=c.N2V_DIMENSIONS,
        window=10,
        min_count=1,
        sg=1,
        workers=1,
        seed=c.GLOBAL_SEED,
    )
    vectors = np.array([model.wv[str(n)] for n in giant_nodes])

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
        method="Metapath-biased random walks (edge_type_c domain-continuity bias) + Word2Vec + UMAP(3D)",
        library_call="custom weighted walk generator -> gensim.models.Word2Vec(sg=1) -> umap.UMAP(3D)",
        params={
            "same_domain_boost": SAME_DOMAIN_BOOST,
            "walk_length": WALK_LENGTH,
            "num_walks": NUM_WALKS,
            "dimensions": c.N2V_DIMENSIONS,
            "seed": c.GLOBAL_SEED,
        },
        weighting_desc=f"walk transitions weighted {SAME_DOMAIN_BOOST}x toward same-domain edge types",
        fallback_notes=(
            f"{len(remaining)} nodes outside the giant component (walks can't leave a "
            f"disconnected component) placed via Fibonacci-sphere shell, "
            f"radius_factor={c.SHELL_RADIUS_FACTOR}"
        ),
    )
    return coords, meta
