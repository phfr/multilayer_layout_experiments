"""Centrality-elevation hybrid: reuse an existing 2D force layout for x/y,
set z from a node centrality metric. Reads x_fa2_1/y_fa2_1 back from the live
nodes.tsv if fa2_1 has already been written (registry runs fa2_1 first); if
not present (e.g. running this layout standalone via --only), recomputes an
equivalent xy fresh and logs that fallback explicitly.
"""

from __future__ import annotations

import math
from collections import Counter

import networkx as nx

import common as c

def _get_or_compute_fa2_xy(G, nodes_path):
    id_col = c.columns(G).id
    header, rows = c.read_nodes_tsv(nodes_path)
    if "x_fa2_1" in header and "y_fa2_1" in header and id_col in header:
        xy = {row[id_col]: (float(row["x_fa2_1"]), float(row["y_fa2_1"])) for row in rows}
        if set(xy) == set(G.nodes()):
            return xy, False
    pos = nx.forceatlas2_layout(G, max_iter=100, seed=c.GLOBAL_SEED, dim=2)
    return {n: (float(p[0]), float(p[1])) for n, p in pos.items()}, True


def _centrality_elevation(G, node_layer, nodes_path, *, metric_name: str, metric_fn, library_call: str | None = None):
    xy, recomputed_xy = _get_or_compute_fa2_xy(G, nodes_path)
    metric = metric_fn(G)

    x_vals = [p[0] for p in xy.values()]
    y_vals = [p[1] for p in xy.values()]
    xy_spread = max(max(x_vals) - min(x_vals), max(y_vals) - min(y_vals))
    m_vals = list(metric.values())
    m_min, m_max = min(m_vals), max(m_vals)
    m_range = (m_max - m_min) or 1.0

    coords = {}
    for n, (x, y) in xy.items():
        z = ((metric[n] - m_min) / m_range - 0.5) * xy_spread
        coords[n] = (x, y, z)

    fallback_notes = f"z = normalized {metric_name}, scaled to xy spread ({xy_spread:.4f})"
    if recomputed_xy:
        fallback_notes += "; x_fa2_1/y_fa2_1 not found in nodes.tsv, recomputed fa2_1-equivalent xy fresh"

    meta = dict(
        method=f"Centrality-elevation hybrid: fa2_1 xy + z from {metric_name}",
        library_call=library_call or f"nx.{metric_name}(G)",
        params={"reused_fa2_1_xy": not recomputed_xy},
        weighting_desc="none",
        fallback_notes=fallback_notes,
    )
    return coords, meta


def run_topodeg_1(G, node_layer, nodes_path):
    return _centrality_elevation(
        G, node_layer, nodes_path,
        metric_name="degree_centrality",
        metric_fn=nx.degree_centrality,
    )


def run_topobet_1(G, node_layer, nodes_path):
    return _centrality_elevation(
        G, node_layer, nodes_path,
        metric_name="betweenness_centrality",
        metric_fn=lambda G: nx.betweenness_centrality(G, normalized=True),
    )


def _domain_touch_count(G: nx.Graph) -> dict[str, int]:
    """# of distinct layers (values of the layer column) a node's own layer
    plus its 1-hop neighborhood collectively touch (1 .. number of layers).
    Catches low-degree nodes that quietly bridge multiple layers -- a signal
    neither the layer label alone nor centrality alone expresses: a node
    adjacent to neighbours from two other layers ranks as a strong connector
    even at degree 2."""
    counts: dict[str, int] = {}
    for n in G.nodes():
        domains = {G.nodes[n]["layer"]}
        for nb in G.neighbors(n):
            domains.add(G.nodes[nb]["layer"])
        counts[n] = len(domains)
    return counts


def run_domaintouch_1(G, node_layer, nodes_path):
    return _centrality_elevation(
        G, node_layer, nodes_path,
        metric_name="domain_touch_count",
        metric_fn=_domain_touch_count,
        library_call="custom: |{layer of node} union {layers of its neighbors}|",
    )


def _rarest_incident_edge_score(G: nx.Graph) -> dict[str, float]:
    """z-score per node = -ln(rarest_incident_edge_type_count / total_edges):
    a node touching a rare edge type (value of the edge-type column) scores
    high even at low degree; a hub whose many edges are all of a common type
    scores low despite high degree -- the opposite signal from
    topodeg_1/topobet_1."""
    type_counts = Counter(data.get("edge_type", "") for _, _, data in G.edges(data=True))
    total = sum(type_counts.values()) or 1
    scores: dict[str, float] = {}
    for n in G.nodes():
        incident_types = [G.edges[n, nb].get("edge_type", "") for nb in G.neighbors(n)]
        if not incident_types:
            scores[n] = 0.0
            continue
        rarest_count = min(type_counts[et] for et in incident_types)
        scores[n] = -math.log(rarest_count / total)
    return scores


def run_raritytouch_1(G, node_layer, nodes_path):
    return _centrality_elevation(
        G, node_layer, nodes_path,
        metric_name="rarest_incident_edge_type_score",
        metric_fn=_rarest_incident_edge_score,
        library_call=f"custom: -ln(min({c.columns(G).edge_type} count among incident edges) / total_edges)",
    )
