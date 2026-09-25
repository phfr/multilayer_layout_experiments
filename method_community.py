"""Louvain community detection + meta-layout + local sublayout per
community. Structurally different grouping principle from shell_*/fa2sep_1
(which use the a priori layer3 label): communities are detected purely from
graph topology (modularity optimization), so a community can span multiple
layers if they're densely interconnected -- reveals emergent structure the
layer-based layouts can't.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import community as community_louvain
import networkx as nx
import numpy as np

import common as c


def run_community_1(G, node_layer, nodes_path):
    partition = community_louvain.best_partition(G, random_state=c.GLOBAL_SEED)
    communities: dict[int, list[str]] = defaultdict(list)
    for node, cid in partition.items():
        communities[cid].append(node)

    Gm = nx.Graph()
    Gm.add_nodes_from(communities.keys())
    for u, v in G.edges():
        cu, cv = partition[u], partition[v]
        if cu == cv:
            continue
        if Gm.has_edge(cu, cv):
            Gm[cu][cv]["weight"] += 1
        else:
            Gm.add_edge(cu, cv, weight=1)

    meta_pos = nx.forceatlas2_layout(Gm, max_iter=100, seed=c.GLOBAL_SEED, dim=3, weight="weight")
    meta_arr = np.array(list(meta_pos.values()))
    meta_spread = float(np.linalg.norm(meta_arr - meta_arr.mean(axis=0), axis=1).mean()) or 1.0

    coords = {}
    n_singletons = 0
    for cid, members in communities.items():
        slot = meta_pos[cid]
        if len(members) == 1:
            coords[members[0]] = slot
            n_singletons += 1
            continue
        sub = G.subgraph(members)
        local = c.local_component_sublayout(sub)
        scale = c.LOCAL_CLUSTER_RADIUS * meta_spread * math.sqrt(len(members)) / math.sqrt(10)
        for node, p in local.items():
            coords[node] = slot + p * scale

    meta = dict(
        method=(
            "Louvain community detection, meta-layout (ForceAtlas2 3D on the community "
            "graph), local sublayout per community"
        ),
        library_call=(
            "community.best_partition(G) -> "
            "nx.forceatlas2_layout(meta_graph, dim=3, weight='weight')"
        ),
        params={"seed": c.GLOBAL_SEED, "n_communities": len(communities)},
        weighting_desc="meta-graph edges weighted by inter-community edge count",
        fallback_notes=(
            f"{n_singletons} singleton communities (isolated nodes) placed exactly at "
            f"their meta-node slot"
        ),
    )
    return coords, meta


def run_community_shellz_1(G, node_layer, nodes_path):
    """Surgical blend: community_1's real xy (force-computed, encodes real
    topology) + shell_1's radius-by-layer3 formula, reused as a signed z
    elevation applied per COMMUNITY's predominant layer3 (not per node). No
    Procrustes alignment is needed -- this reuses shell_1's radius FORMULA,
    not its literal coordinates, so there's no coordinate-frame mismatch to
    resolve. Result: "communities sorted by predominant measurement layer",
    a read neither community_1 (layer-blind) nor shell_1 (community-blind)
    gives alone.
    """
    community_coords, _ = run_community_1(G, node_layer, nodes_path)

    partition = community_louvain.best_partition(G, random_state=c.GLOBAL_SEED)
    members_by_community: dict[int, list[str]] = defaultdict(list)
    for n, cid in partition.items():
        members_by_community[cid].append(n)

    layer_counts = Counter(node_layer.values())
    protein_r = c.SHELL_R_BASE * math.sqrt(layer_counts.get("protein", 1))
    metabolite_r = c.SHELL_R_BASE * math.sqrt(layer_counts.get("metabolite", 1))
    z_by_layer = {"protein": -protein_r, "bridge": 0.0, "metabolite": metabolite_r}

    n_pure = 0
    n_mixed = 0
    coords = {}
    for members in members_by_community.values():
        layer_votes = Counter(node_layer[n] for n in members)
        predominant, top_count = layer_votes.most_common(1)[0]
        if top_count == len(members):
            n_pure += 1
        else:
            n_mixed += 1
        z = z_by_layer[predominant]
        for n in members:
            x, y, _ = community_coords[n]
            coords[n] = (x, y, z)

    meta = dict(
        method=(
            "Surgical blend: community_1's xy (real force-computed topology) + "
            "shell_1's radius-by-layer3 formula reused as a signed z elevation, "
            "applied per COMMUNITY's predominant layer3 (not per node)"
        ),
        library_call="community.best_partition(G) -> mode(layer3) per community -> z_by_layer lookup",
        params={
            "z_by_layer": {k: round(v, 4) for k, v in z_by_layer.items()},
            "seed": c.GLOBAL_SEED,
        },
        weighting_desc="none",
        fallback_notes=(
            f"{len(members_by_community)} communities: {n_pure} single-layer3-only, "
            f"{n_mixed} mixed-layer3 (z assigned by majority vote -- a lossy "
            f"simplification for genuinely mixed communities, which is the whole "
            f"point of Louvain finding them)"
        ),
    )
    return coords, meta
