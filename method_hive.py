"""Hive-plot-inspired multi-axis layouts: one radial spoke per group, nodes
placed along their group's axis ordered by degree (hubs near the shared
origin, low-degree/isolated nodes toward the tip). Pure deterministic
geometry, no simulation -- inspired by Hive Plots (Krzywinski, hiveplot.com),
which place typed nodes on axes ordered by a numeric attribute specifically
to cut hairball clutter in typed networks. Adapted from their usual 2D
layout to axes spread in 3D via a Fibonacci-sphere direction.

hive3_1 groups by layer (one axis per --layer-order entry); hive5_1
groups by the raw value of the type column (--type-column; one axis per
distinct value, ordered by the value's layer then name) -- a finer split
that shows sub-types within a layer (e.g. differentially-expressed vs.
bridge proteins) on their own spoke. The base names keep their historical
3/5 suffixes; the axis count is data-driven.
"""

from __future__ import annotations

import common as c


def _hive_layout(G, groups: dict[str, list[str]]):
    directions = dict(zip(groups, c.fibonacci_sphere(len(groups))))
    coords = {}
    axis_sizes = {}
    for group, group_nodes in groups.items():
        nodes_sorted = sorted(group_nodes, key=lambda n: (-G.degree(n), c.node_key(n)))
        n_k = len(nodes_sorted)
        axis_sizes[group] = n_k
        direction = directions[group]
        for rank, nid in enumerate(nodes_sorted):
            frac = (rank + 1) / n_k
            coords[nid] = direction * (c.HIVE_AXIS_LENGTH * frac)
    return coords, axis_sizes


def run_hive3_1(G, node_layer, nodes_path):
    layers = c.layer_order(G)
    groups = {layer: [n for n, l in node_layer.items() if l == layer] for layer in layers}
    coords, axis_sizes = _hive_layout(G, groups)

    meta = dict(
        method=f"Hive-plot-inspired {len(layers)}-axis layout: one radial spoke per layer, ordered by degree",
        library_call=f"deterministic geometry (no simulation); axis directions from fibonacci_sphere({len(layers)})",
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
    type_col = c.columns(G).type
    layer_rank = {layer: i for i, layer in enumerate(c.layer_order(G))}
    by_type: dict[str, list[str]] = {}
    for n in G.nodes():
        by_type.setdefault(G.nodes[n].get("type", ""), []).append(n)
    # Axes ordered by the layer their nodes belong to (then name), so sub-types
    # of one layer sit on adjacent spokes.
    type_values = sorted(
        by_type, key=lambda ta: (min(layer_rank[node_layer[n]] for n in by_type[ta]), ta)
    )
    groups = {ta: by_type[ta] for ta in type_values}
    coords, axis_sizes = _hive_layout(G, groups)

    meta = dict(
        method=(
            f"Hive-plot-inspired {len(groups)}-axis layout: one radial spoke per raw {type_col} "
            f"value ({'/'.join(type_values)}), ordered by degree -- shows sub-types "
            "within a layer on their own spoke"
        ),
        library_call=f"deterministic geometry (no simulation); axis directions from fibonacci_sphere({len(groups)})",
        params={
            "axis_length": c.HIVE_AXIS_LENGTH,
            "axis_sizes": axis_sizes,
            "order": "degree desc (hubs near shared origin)",
        },
        weighting_desc="none",
        fallback_notes=f"isolated (degree-0) nodes land at the tip of their {type_col} value's axis (lowest rank)",
    )
    return coords, meta
