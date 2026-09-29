"""ForceAtlas2-based layouts: shared xy + discrete z-stack per layer (planes
ordered by --layer-order), per-layer independent runs, and native 3D. Small (2-10 node) connected
components are left in the force simulation (force-directed layouts handle
them fine); only fully isolated (degree-0) nodes are excluded and placed via
a deterministic ring/shell afterward.
"""

from __future__ import annotations

import math
from collections import Counter

import networkx as nx

import common as c


def _rarity_weights(G: nx.Graph) -> dict[str, float]:
    """weight = -ln(edge_type_c count / total edges): one weight per distinct
    edge_type_c value (data-driven), not the binary within/cross-layer split
    fa2_3 uses -- edges of a rare type get pulled tighter than edges of a
    common type instead of both being lumped into one 'cross-layer' bucket."""
    counts = Counter(data.get("edge_type_c", "") for _, _, data in G.edges(data=True))
    total = sum(counts.values()) or 1
    return {et: -math.log(cnt / total) for et, cnt in counts.items()}


def run_fa2_1(G, node_layer, nodes_path):
    iso = c.isolated_nodes(G)
    Gc = G.subgraph([n for n in G.nodes() if n not in iso])
    xy = nx.forceatlas2_layout(Gc, max_iter=100, seed=c.GLOBAL_SEED, dim=2)
    coords, z_by_layer = c.layer_z_stack(xy, node_layer, order=c.layer_order(G))
    coords.update(c.place_isolated_ring_per_layer(xy, node_layer, iso, z_by_layer))
    meta = dict(
        method="ForceAtlas2 (shared xy, discrete z per layer)",
        library_call="nx.forceatlas2_layout(G, max_iter=100, seed=42, dim=2)",
        params={"max_iter": 100, "seed": c.GLOBAL_SEED, "dim": 2},
        weighting_desc="unweighted",
        fallback_notes=f"{len(iso)} isolated nodes ring-placed per layer at z={z_by_layer}",
    )
    return coords, meta


def run_fa2_2(G, node_layer, nodes_path):
    iso = c.isolated_nodes(G)
    Gc = G.subgraph([n for n in G.nodes() if n not in iso])
    xy = nx.forceatlas2_layout(
        Gc, max_iter=100, seed=c.GLOBAL_SEED, dim=2, linlog=True, dissuade_hubs=True
    )
    coords, z_by_layer = c.layer_z_stack(xy, node_layer, order=c.layer_order(G))
    coords.update(c.place_isolated_ring_per_layer(xy, node_layer, iso, z_by_layer))
    meta = dict(
        method="ForceAtlas2 (linlog + dissuade_hubs, shared xy, discrete z per layer)",
        library_call=(
            "nx.forceatlas2_layout(G, max_iter=100, seed=42, dim=2, "
            "linlog=True, dissuade_hubs=True)"
        ),
        params={
            "max_iter": 100,
            "seed": c.GLOBAL_SEED,
            "dim": 2,
            "linlog": True,
            "dissuade_hubs": True,
        },
        weighting_desc="unweighted",
        fallback_notes=f"{len(iso)} isolated nodes ring-placed per layer at z={z_by_layer}",
    )
    return coords, meta


def run_fa2_3(G, node_layer, nodes_path):
    iso = c.isolated_nodes(G)
    Gc = G.subgraph([n for n in G.nodes() if n not in iso]).copy()
    c.set_attraction_weight(Gc, node_layer)
    xy = nx.forceatlas2_layout(Gc, max_iter=100, seed=c.GLOBAL_SEED, dim=2, weight="fa_weight")
    coords, z_by_layer = c.layer_z_stack(xy, node_layer, order=c.layer_order(G))
    coords.update(c.place_isolated_ring_per_layer(xy, node_layer, iso, z_by_layer))
    meta = dict(
        method="ForceAtlas2 (cross-layer down-weighted, shared xy, discrete z per layer)",
        library_call="nx.forceatlas2_layout(G, max_iter=100, seed=42, dim=2, weight='fa_weight')",
        params={"max_iter": 100, "seed": c.GLOBAL_SEED, "dim": 2, "weight": "fa_weight"},
        weighting_desc=f"attraction: within={c.ATTRACTION_WITHIN}, cross={c.ATTRACTION_CROSS}",
        fallback_notes=f"{len(iso)} isolated nodes ring-placed per layer at z={z_by_layer}",
    )
    return coords, meta


def run_fa2sep_1(G, node_layer, nodes_path):
    layers = c.layer_order(G)
    xy: dict[str, tuple] = {}
    iso_all: list[str] = []
    for layer in layers:
        layer_nodes = [n for n, l in node_layer.items() if l == layer]
        sub = G.subgraph(layer_nodes)
        iso = c.isolated_nodes(sub)
        iso_all.extend(iso)
        sub_conn = sub.subgraph([n for n in layer_nodes if n not in iso])
        if sub_conn.number_of_nodes():
            xy.update(nx.forceatlas2_layout(sub_conn, max_iter=100, seed=c.GLOBAL_SEED, dim=2))

    coords, z_by_layer = c.layer_z_stack(xy, node_layer, order=c.layer_order(G))
    coords.update(c.place_isolated_ring_per_layer(xy, node_layer, iso_all, z_by_layer))
    meta = dict(
        method=(
            "ForceAtlas2 run independently per layer (induced subgraphs), "
            "xy left overlapping across layers, z-stacked"
        ),
        library_call=f"nx.forceatlas2_layout(layer_subgraph, max_iter=100, seed=42, dim=2) x{len(layers)}",
        params={"max_iter": 100, "seed": c.GLOBAL_SEED, "dim": 2, "layers": layers},
        weighting_desc="none (cross-layer edges excluded by induced subgraph)",
        fallback_notes=(
            f"{len(iso_all)} within-layer-isolated nodes ring-placed per their own "
            f"layer at z={z_by_layer}"
        ),
    )
    return coords, meta


def run_fa2spectral_1(G, node_layer, nodes_path):
    """Spectral-seeded ForceAtlas2 (the fCoSE trick from Cytoscape's
    Compound Spring Embedder: seed force-directed layout with a spectral
    embedding instead of random init, for better global structure / faster
    convergence). Shared xy, discrete z per layer like fa2_1.
    """
    iso = c.isolated_nodes(G)
    Gc = G.subgraph([n for n in G.nodes() if n not in iso])
    # Rescale + jitter the seed: raw spectral coordinates put structurally
    # equivalent nodes at identical points, which makes FA2's repulsion
    # explode (see common.SEED_JITTER_FRACTION).
    seed_pos = c.prepare_seed_positions(c.spectral_positions(Gc, 2))
    xy = nx.forceatlas2_layout(Gc, pos=seed_pos, max_iter=100, seed=c.GLOBAL_SEED, dim=2)
    coords, z_by_layer = c.layer_z_stack(xy, node_layer, order=c.layer_order(G))
    coords.update(c.place_isolated_ring_per_layer(xy, node_layer, iso, z_by_layer))
    meta = dict(
        method="ForceAtlas2 seeded with a spectral embedding (fCoSE-style warm start), z-stacked",
        library_call=(
            "nx.forceatlas2_layout(G, pos=prepare_seed_positions(spectral_positions(G, 2)), "
            "max_iter=100, seed=42, dim=2)"
        ),
        params={
            "max_iter": 100,
            "seed": c.GLOBAL_SEED,
            "dim": 2,
            "seeded": True,
            "seed_rms_per_sqrt_n": c.SEED_TARGET_RMS_PER_SQRT_N,
            "seed_jitter_fraction": c.SEED_JITTER_FRACTION,
        },
        weighting_desc="unweighted",
        fallback_notes=f"{len(iso)} isolated nodes ring-placed per layer at z={z_by_layer}",
    )
    return coords, meta


def run_fa23d_1(G, node_layer, nodes_path):
    iso = c.isolated_nodes(G)
    Gc = G.subgraph([n for n in G.nodes() if n not in iso])
    pos = nx.forceatlas2_layout(Gc, max_iter=100, seed=c.GLOBAL_SEED, dim=3)
    coords = {n: p for n, p in pos.items()}
    coords.update(c.place_isolated_sphere_shell(pos, iso))
    meta = dict(
        method="ForceAtlas2 native 3D",
        library_call="nx.forceatlas2_layout(G, max_iter=100, seed=42, dim=3)",
        params={"max_iter": 100, "seed": c.GLOBAL_SEED, "dim": 3},
        weighting_desc="unweighted",
        fallback_notes=f"{len(iso)} isolated nodes placed on a Fibonacci-sphere shell",
    )
    return coords, meta


def run_fa23d_2(G, node_layer, nodes_path):
    iso = c.isolated_nodes(G)
    Gc = G.subgraph([n for n in G.nodes() if n not in iso]).copy()
    c.set_attraction_weight(Gc, node_layer)
    pos = nx.forceatlas2_layout(Gc, max_iter=100, seed=c.GLOBAL_SEED, dim=3, weight="fa_weight")
    coords = {n: p for n, p in pos.items()}
    coords.update(c.place_isolated_sphere_shell(pos, iso))
    meta = dict(
        method="ForceAtlas2 native 3D, cross-layer down-weighted",
        library_call="nx.forceatlas2_layout(G, max_iter=100, seed=42, dim=3, weight='fa_weight')",
        params={"max_iter": 100, "seed": c.GLOBAL_SEED, "dim": 3, "weight": "fa_weight"},
        weighting_desc=f"attraction: within={c.ATTRACTION_WITHIN}, cross={c.ATTRACTION_CROSS}",
        fallback_notes=f"{len(iso)} isolated nodes placed on a Fibonacci-sphere shell",
    )
    return coords, meta


def run_fa2rarity_1(G, node_layer, nodes_path):
    iso = c.isolated_nodes(G)
    Gc = G.subgraph([n for n in G.nodes() if n not in iso]).copy()
    weights = _rarity_weights(Gc)
    for _, _, data in Gc.edges(data=True):
        data["rarity_weight"] = weights.get(data.get("edge_type_c", ""), 1.0)
    xy = nx.forceatlas2_layout(Gc, max_iter=100, seed=c.GLOBAL_SEED, dim=2, weight="rarity_weight")
    coords, z_by_layer = c.layer_z_stack(xy, node_layer, order=c.layer_order(G))
    coords.update(c.place_isolated_ring_per_layer(xy, node_layer, iso, z_by_layer))
    meta = dict(
        method=(
            "ForceAtlas2 with per-edge_type_c rarity weighting "
            "(-ln(count/total), not the binary within/cross-layer split), z-stacked"
        ),
        library_call="nx.forceatlas2_layout(G, max_iter=100, seed=42, dim=2, weight='rarity_weight')",
        params={
            "max_iter": 100,
            "seed": c.GLOBAL_SEED,
            "dim": 2,
            "rarity_weights": {k: round(v, 3) for k, v in weights.items()},
        },
        weighting_desc="attraction weight = -ln(edge_type_c count / total edges), one bucket per distinct type",
        fallback_notes=f"{len(iso)} isolated nodes ring-placed per layer at z={z_by_layer}",
    )
    return coords, meta
