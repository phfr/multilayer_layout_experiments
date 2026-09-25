"""PAGA-style two-level coarse-then-fine layout (Wolf et al., "PAGA: graph
abstraction reconciles clustering with trajectory inference through a
topology preserving map of single cells", Genome Biology 2019). Build an
abstracted graph of Louvain communities, weight each meta-edge by how much
MORE connected two communities are than a configuration-model null expects
(not just raw cross-community edge count), lay that small graph out first,
then initialize every node at its community's coarse position (with a small
within-community offset so nodes don't start exactly coincident) and refine
with a full spring-layout pass over the real graph. Distinct from
`community_1`: here the community structure drives INITIALIZATION for a
real physics refinement over all edges, not a fixed placement.
"""

from __future__ import annotations

import math
from collections import defaultdict

import community as community_louvain
import networkx as nx
import numpy as np

import common as c


def _paga_meta_graph(G: nx.Graph, partition: dict[str, int]) -> nx.Graph:
    degree = dict(G.degree())
    total_edges = G.number_of_edges()
    community_degree_sum: dict[int, float] = defaultdict(float)
    for n, cid in partition.items():
        community_degree_sum[cid] += degree[n]

    observed: dict[tuple[int, int], int] = defaultdict(int)
    for u, v in G.edges():
        cu, cv = partition[u], partition[v]
        if cu == cv:
            continue
        key = (cu, cv) if cu < cv else (cv, cu)
        observed[key] += 1

    Gm = nx.Graph()
    Gm.add_nodes_from(set(partition.values()))
    for (cu, cv), obs in observed.items():
        expected = (community_degree_sum[cu] * community_degree_sum[cv]) / (2.0 * total_edges)
        if expected <= 0:
            continue
        weight = obs / expected
        Gm.add_edge(cu, cv, weight=weight)
    return Gm


def run_paga_1(G, node_layer, nodes_path):
    partition = community_louvain.best_partition(G, random_state=c.GLOBAL_SEED)
    communities: dict[int, list[str]] = defaultdict(list)
    for n, cid in partition.items():
        communities[cid].append(n)

    Gm = _paga_meta_graph(G, partition)
    meta_pos = nx.forceatlas2_layout(Gm, max_iter=100, seed=c.GLOBAL_SEED, dim=3, weight="weight")
    meta_arr = np.array(list(meta_pos.values()))
    meta_spread = float(np.linalg.norm(meta_arr - meta_arr.mean(axis=0), axis=1).mean()) or 1.0

    # Initialize every node near its community's coarse position, with a
    # small within-community Fibonacci-sphere offset so same-community
    # nodes don't start exactly coincident (spring_layout needs distinct
    # starting points to have a well-defined initial gradient).
    init_pos: dict[str, np.ndarray] = {}
    offset_scale = c.LOCAL_CLUSTER_RADIUS * meta_spread
    for cid, members in communities.items():
        slot = meta_pos[cid]
        members_sorted = sorted(members, key=lambda n: int(n))
        dirs = c.fibonacci_sphere(len(members_sorted))
        for nid, d in zip(members_sorted, dirs):
            init_pos[nid] = slot + d * offset_scale * math.sqrt(len(members_sorted)) / math.sqrt(10)

    fine_pos = nx.spring_layout(G, pos=init_pos, dim=3, seed=c.GLOBAL_SEED, iterations=50)
    coords = {n: p for n, p in fine_pos.items()}

    meta = dict(
        method=(
            "PAGA-style two-level layout: Louvain communities abstracted into a "
            "meta-graph (edges weighted by observed/expected cross-community "
            "connectivity vs. a configuration-model null), meta-graph laid out with "
            "ForceAtlas2, then every node initialized near its community's coarse "
            "position and refined with a full spring-layout pass over the real graph"
        ),
        library_call=(
            "nx.forceatlas2_layout(meta_graph, dim=3, weight='weight') -> "
            "nx.spring_layout(G, pos=init_pos, dim=3, seed=42, iterations=50)"
        ),
        params={
            "n_communities": len(communities),
            "seed": c.GLOBAL_SEED,
            "meta_iterations": 100,
            "fine_iterations": 50,
        },
        weighting_desc="meta-graph edges weighted by observed/expected cross-community connectivity (configuration-model null)",
        fallback_notes=(
            "no separate fallback needed: isolated nodes form singleton communities, "
            "get their own meta-graph slot (placed via FA2's native disconnected-graph "
            "handling), and spring_layout refines every node including isolated ones"
        ),
    )
    return coords, meta
