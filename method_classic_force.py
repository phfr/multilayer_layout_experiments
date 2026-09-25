"""Classic distance/force layouts needing explicit disconnected-graph handling:
Fruchterman-Reingold (nx.spring_layout) and Kamada-Kawai.

Fruchterman-Reingold: same isolated-only exclusion as the ForceAtlas2 family
(small multi-node components are left in, force-directed handles them fine).

Kamada-Kawai: relies on all-pairs shortest paths, which is only meaningful
within a single connected component. Runs on the giant component (745/900
nodes) only; every other node (isolated singles + the 14 small components)
is placed via common.deterministic_fallback_shell.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

import common as c


def run_fr3d_1(G, node_layer, nodes_path):
    iso = c.isolated_nodes(G)
    Gc = G.subgraph([n for n in G.nodes() if n not in iso])
    pos = nx.spring_layout(Gc, dim=3, seed=c.GLOBAL_SEED)
    coords = {n: p for n, p in pos.items()}
    coords.update(c.place_isolated_sphere_shell(pos, iso))
    meta = dict(
        method="Fruchterman-Reingold / spring, native 3D",
        library_call="nx.spring_layout(G, dim=3, seed=42)",
        params={"dim": 3, "seed": c.GLOBAL_SEED},
        weighting_desc="unweighted",
        fallback_notes=f"{len(iso)} isolated nodes placed on a Fibonacci-sphere shell",
    )
    return coords, meta


def run_fr3d_2(G, node_layer, nodes_path):
    iso = c.isolated_nodes(G)
    Gc = G.subgraph([n for n in G.nodes() if n not in iso]).copy()
    c.set_attraction_weight(Gc, node_layer)
    pos = nx.spring_layout(Gc, dim=3, seed=c.GLOBAL_SEED, weight="fa_weight")
    coords = {n: p for n, p in pos.items()}
    coords.update(c.place_isolated_sphere_shell(pos, iso))
    meta = dict(
        method="Fruchterman-Reingold / spring, native 3D, cross-layer down-weighted",
        library_call="nx.spring_layout(G, dim=3, seed=42, weight='fa_weight')",
        params={"dim": 3, "seed": c.GLOBAL_SEED, "weight": "fa_weight"},
        weighting_desc=f"attraction: within={c.ATTRACTION_WITHIN}, cross={c.ATTRACTION_CROSS}",
        fallback_notes=f"{len(iso)} isolated nodes placed on a Fibonacci-sphere shell",
    )
    return coords, meta


def _kk3d(G, node_layer, *, weighted: bool):
    giant = c.giant_component_nodes(G)
    giant_nodes = sorted(giant, key=lambda n: int(n))
    sub = G.subgraph(giant_nodes).copy()

    weight_attr = None
    weighting_desc = "unweighted (hop distance)"
    if weighted:
        c.set_distance_weight(sub, node_layer)
        weight_attr = "kk_weight"
        weighting_desc = f"distance: within={c.DISTANCE_WITHIN}, cross={c.DISTANCE_CROSS}"

    np.random.seed(c.GLOBAL_SEED)  # kamada_kawai_layout(dim=3) seeds its own init from this
    giant_pos = nx.kamada_kawai_layout(sub, dim=3, weight=weight_attr)

    remaining = [n for n in G.nodes() if n not in giant]
    fallback_pos = c.deterministic_fallback_shell(giant_pos, remaining, G)

    coords = {n: p for n, p in giant_pos.items()}
    coords.update(fallback_pos)

    meta = dict(
        method=(
            f"Kamada-Kawai on giant component ({len(giant_nodes)} nodes) + "
            f"deterministic fallback shell for {len(remaining)} remaining nodes"
        ),
        library_call="nx.kamada_kawai_layout(giant_subgraph, dim=3, weight=...)",
        params={"dim": 3, "weight": weight_attr},
        weighting_desc=weighting_desc,
        fallback_notes=(
            f"{len(remaining)} nodes outside the giant component (isolated + small "
            f"components) placed via Fibonacci-sphere shell, radius_factor="
            f"{c.SHELL_RADIUS_FACTOR}"
        ),
    )
    return coords, meta


def run_kk3d_1(G, node_layer, nodes_path):
    return _kk3d(G, node_layer, weighted=False)


def run_kk3d_2(G, node_layer, nodes_path):
    return _kk3d(G, node_layer, weighted=True)
