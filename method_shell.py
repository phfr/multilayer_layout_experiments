"""Concentric shell/sphere layout: protein/bridge/metabolite each get their
own spherical shell (radius scaled by node count), with nodes placed on a
deterministic Fibonacci-sphere spiral -- no force simulation, so it can't
fail or explode on the 118-component graph. Nodes are ordered within each
layer by (connected-component rank, then id) so same-component nodes land
angularly close together.
"""

from __future__ import annotations

import math
from collections import Counter

import common as c

# Bands over the rarest edge_type_c a node touches: (0,50) covers the 3
# rarest of the 10 known edge types (metabolite-protein=28,
# metabolite-transcript=34, transcript-bridge=43); (50,120) covers the
# bridge-touching mid-rarity types (107/110/119); (120,200) covers
# protein-transcript/protein-protein (152/187); (200,inf) covers only nodes
# whose ALL edges are one of the two most common types
# (transcript-transcript=409, metabolite-metabolite=500). Isolated nodes (no
# incident edges) get their own outermost band.
RARITY_BANDS = ((0, 50), (50, 120), (120, 200), (200, float("inf")))


def _shell_layout(G, node_layer, *, order: tuple[str, str, str]):
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
    order = ("protein", "bridge", "metabolite")
    coords, radii = _shell_layout(G, node_layer, order=order)
    meta = dict(
        method="Concentric shell layout: protein (inner) / bridge (mid) / metabolite (outer)",
        library_call="deterministic Fibonacci-sphere placement per layer3 (no simulation)",
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
    order = ("metabolite", "bridge", "protein")
    coords, radii = _shell_layout(G, node_layer, order=order)
    meta = dict(
        method="Concentric shell layout: metabolite (inner) / bridge (mid) / protein (outer)",
        library_call="deterministic Fibonacci-sphere placement per layer3 (no simulation)",
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
    assignment is by rarity band (the rarest edge_type_c a node touches),
    not layer3 -- a self-contained new grouping principle: rare cross-domain
    connectors form the innermost shell regardless of which layer they
    belong to, common-only-touching nodes form the outer shells."""
    type_counts = Counter(data.get("edge_type_c", "") for _, _, data in G.edges(data=True))

    def rarest_count(n):
        incident = [G.edges[n, nb].get("edge_type_c", "") for nb in G.neighbors(n)]
        if not incident:
            return None
        return min(type_counts[et] for et in incident)

    n_bands = len(RARITY_BANDS) + 1  # + 1 for isolated nodes' own outermost band
    band_of: dict[str, int] = {}
    for n in G.nodes():
        rc = rarest_count(n)
        if rc is None:
            band_of[n] = n_bands - 1
            continue
        for i, (lo, hi) in enumerate(RARITY_BANDS):
            if lo <= rc < hi:
                band_of[n] = i
                break

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
            "touches (innermost = touches a rare cross-domain type), not layer3"
        ),
        library_call="deterministic Fibonacci-sphere placement per rarity band (no simulation)",
        params={
            "bands": RARITY_BANDS,
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
