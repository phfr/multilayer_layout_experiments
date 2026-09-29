"""Concentric shell/sphere layout: each layer (in --layer-order, first =
innermost) gets its own spherical shell (radius scaled by node count), with
nodes placed on a deterministic Fibonacci-sphere spiral -- no force
simulation, so it can't fail or explode on a many-component graph. Nodes
are ordered within each layer by (connected-component rank, then id) so
same-component nodes land angularly close together.
"""

from __future__ import annotations

import math
from collections import Counter

import common as c


def _shell_layout(G, node_layer, *, order: tuple[str, ...]):
    coords = {}
    prev_r = 0.0
    radii = {}
    for layer in order:
        layer_nodes = [n for n, l in node_layer.items() if l == layer]
        n_k = len(layer_nodes)
        r_k = max(c.SHELL_R_BASE * math.sqrt(max(n_k, 1)), prev_r * c.SHELL_MIN_GAP_FACTOR)
        radii[layer] = r_k
        prev_r = r_k

        sub = G.subgraph(layer_nodes)
        comps = c.sorted_components(sub) if n_k else []
        ordered_nodes: list[str] = []
        for comp in comps:
            ordered_nodes.extend(sorted(comp, key=lambda x: int(x)))

        dirs = c.fibonacci_sphere(len(ordered_nodes))
        for nid, d in zip(ordered_nodes, dirs):
            coords[nid] = d * r_k
    return coords, radii


def run_shell_1(G, node_layer, nodes_path):
    order = c.layer_order(G)
    coords, radii = _shell_layout(G, node_layer, order=order)
    meta = dict(
        method=f"Concentric shell layout, inner -> outer: {' / '.join(order)}",
        library_call="deterministic Fibonacci-sphere placement per layer (no simulation)",
        params={
            "order": order,
            "shell_r_base": c.SHELL_R_BASE,
            "shell_min_gap_factor": c.SHELL_MIN_GAP_FACTOR,
            "radii": {k: round(v, 4) for k, v in radii.items()},
        },
        weighting_desc="none",
        fallback_notes=(
            "node order within each shell: (connected-component rank, then id) so "
            "same-component nodes are angularly contiguous; handles isolated nodes "
            "naturally as trivial 1-node components"
        ),
    )
    return coords, meta


def run_shell_2(G, node_layer, nodes_path):
    order = tuple(reversed(c.layer_order(G)))
    coords, radii = _shell_layout(G, node_layer, order=order)
    meta = dict(
        method=f"Concentric shell layout (reversed), inner -> outer: {' / '.join(order)}",
        library_call="deterministic Fibonacci-sphere placement per layer (no simulation)",
        params={
            "order": order,
            "shell_r_base": c.SHELL_R_BASE,
            "shell_min_gap_factor": c.SHELL_MIN_GAP_FACTOR,
            "radii": {k: round(v, 4) for k, v in radii.items()},
        },
        weighting_desc="none",
        fallback_notes="reversed radius order variant of shell_1; same node-ordering rule",
    )
    return coords, meta


def run_shell_rarity_1(G, node_layer, nodes_path):
    """Same deterministic-geometry mechanism as shell_1/shell_2, but shell
    assignment is by the rarest edge_type_c a node touches, not its layer --
    a self-contained grouping principle: rare cross-domain connectors form
    the innermost shell regardless of which layer they belong to, nodes
    touching only common edge types form the outer shells. Bands are
    data-driven: one per distinct edge_type_c frequency, rarest first (edge
    types with equal counts share a band); isolated nodes (no incident
    edges) get their own outermost band."""
    type_counts = Counter(data.get("edge_type_c", "") for _, _, data in G.edges(data=True))
    distinct_counts = sorted(set(type_counts.values()))
    band_of_count = {cnt: i for i, cnt in enumerate(distinct_counts)}
    bands = {
        i: sorted(et for et, cnt in type_counts.items() if cnt == count)
        for count, i in band_of_count.items()
    }

    def rarest_count(n):
        incident = [G.edges[n, nb].get("edge_type_c", "") for nb in G.neighbors(n)]
        if not incident:
            return None
        return min(type_counts[et] for et in incident)

    n_bands = len(distinct_counts) + 1  # + 1 for isolated nodes' own outermost band
    band_of: dict[str, int] = {}
    for n in G.nodes():
        rc = rarest_count(n)
        band_of[n] = n_bands - 1 if rc is None else band_of_count[rc]

    coords = {}
    prev_r = 0.0
    radii = {}
    band_sizes = {}
    for band in range(n_bands):
        band_nodes = [n for n, b in band_of.items() if b == band]
        n_k = len(band_nodes)
        band_sizes[band] = n_k
        if n_k == 0:
            continue
        r_k = max(c.SHELL_R_BASE * math.sqrt(n_k), prev_r * c.SHELL_MIN_GAP_FACTOR)
        radii[band] = r_k
        prev_r = r_k

        sub = G.subgraph(band_nodes)
        comps = c.sorted_components(sub)
        ordered_nodes: list[str] = []
        for comp in comps:
            ordered_nodes.extend(sorted(comp, key=lambda x: int(x)))

        dirs = c.fibonacci_sphere(len(ordered_nodes))
        for nid, d in zip(ordered_nodes, dirs):
            coords[nid] = d * r_k

    meta = dict(
        method=(
            "Concentric shells by rarity band: shell = rarest edge_type_c a node "
            "touches (innermost = touches the rarest type), not layer"
        ),
        library_call="deterministic Fibonacci-sphere placement per rarity band (no simulation)",
        params={
            "bands": {i: f"{types} (count={distinct_counts[i]})" for i, types in bands.items()},
            "shell_r_base": c.SHELL_R_BASE,
            "shell_min_gap_factor": c.SHELL_MIN_GAP_FACTOR,
            "band_sizes": band_sizes,
            "radii": {k: round(v, 4) for k, v in radii.items()},
        },
        weighting_desc="none",
        fallback_notes=(
            "isolated (degree-0) nodes get their own outermost band; node order within "
            "each band: (connected-component rank, then id)"
        ),
    )
    return coords, meta
