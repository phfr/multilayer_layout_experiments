"""Hive-plot-inspired multi-axis layouts: one radial spoke per group, nodes
placed along their group's axis ordered by degree (hubs near the shared
origin, low-degree/isolated nodes toward the tip). Pure deterministic
geometry, no simulation -- inspired by Hive Plots (Krzywinski, hiveplot.com),
which place typed nodes on axes ordered by a numeric attribute specifically
to cut hairball clutter in typed networks. Adapted from their usual 2D
layout to axes spread in 3D via a Fibonacci-sphere direction.

hive3_1 groups by layer3 (3 axes); hive5_1 groups by the raw type_a (5
axes) -- this makes the 21 dual-typed 'protein,transcript' nodes
structurally visible on their own spoke instead of folded into "protein".
"""

from __future__ import annotations

import common as c


def _hive_layout(G, groups: dict[str, list[str]]):
    directions = dict(zip(groups, c.fibonacci_sphere(len(groups))))
    coords = {}
    axis_sizes = {}
    for group, group_nodes in groups.items():
        nodes_sorted = sorted(group_nodes, key=lambda n: (-G.degree(n), int(n)))
        n_k = len(nodes_sorted)
        axis_sizes[group] = n_k
        direction = directions[group]
        for rank, nid in enumerate(nodes_sorted):
            frac = (rank + 1) / n_k
            coords[nid] = direction * (c.HIVE_AXIS_LENGTH * frac)
    return coords, axis_sizes


def run_hive3_1(G, node_layer, nodes_path):
    layers = ("protein", "bridge", "metabolite")
    groups = {layer: [n for n, l in node_layer.items() if l == layer] for layer in layers}
    coords, axis_sizes = _hive_layout(G, groups)

    meta = dict(
        method="Hive-plot-inspired 3-axis layout: one radial spoke per layer3, ordered by degree",
        library_call="deterministic geometry (no simulation); axis directions from fibonacci_sphere(3)",
        params={
            "axis_length": c.HIVE_AXIS_LENGTH,
            "axis_sizes": axis_sizes,
            "order": "degree desc (hubs near shared origin)",
        },
        weighting_desc="none",
        fallback_notes="isolated (degree-0) nodes land at the tip of their layer's axis (lowest rank)",
    )
    return coords, meta


def run_hive5_1(G, node_layer, nodes_path):
    type_a_values = ("protein", "transcript", "protein,transcript", "bridge", "metabolite")
    groups = {
        ta: [n for n in G.nodes() if G.nodes[n].get("type_a") == ta] for ta in type_a_values
    }
    coords, axis_sizes = _hive_layout(G, groups)

    meta = dict(
        method=(
            "Hive-plot-inspired 5-axis layout: one radial spoke per raw type_a value "
            "(protein/transcript/protein,transcript/bridge/metabolite), ordered by degree. "
            "Makes the 21 dual-typed nodes structurally visible on their own spoke "
            "instead of folded into layer3's coarser 'protein' bucket."
        ),
        library_call="deterministic geometry (no simulation); axis directions from fibonacci_sphere(5)",
        params={
            "axis_length": c.HIVE_AXIS_LENGTH,
            "axis_sizes": axis_sizes,
            "order": "degree desc (hubs near shared origin)",
        },
        weighting_desc="none",
        fallback_notes="isolated (degree-0) nodes land at the tip of their type_a's axis (lowest rank)",
    )
    return coords, meta
