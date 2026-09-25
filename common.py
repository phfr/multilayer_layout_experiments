"""Shared IO, graph-building, weighting, and placement helpers for the ppicml
layout-generation scripts (run_layouts.py + method_*.py).

nodes.tsv / edges.tsv are read as raw strings; only numeric parsing happens
where a specific script needs it. Values are written back with CRLF line
endings to match the existing files' convention.
"""

from __future__ import annotations

import csv
import math
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------
# Constants (all runs record their actual values into layout_run_log.txt)
# --------------------------------------------------------------------------

GLOBAL_SEED = 42

# fa2_1/fa2_2/fa2_3/fa2sep_1: z offset per layer3, as a fraction of the xy spread.
Z_STACK_FRACTION = 0.15

# Attraction-style weighting (ForceAtlas2, spring): higher = stronger pull = closer.
ATTRACTION_WITHIN = 1.0
ATTRACTION_CROSS = 0.2

# Distance-style weighting (Kamada-Kawai, MDS hop length): higher = target distance farther.
DISTANCE_WITHIN = 1.0
DISTANCE_CROSS = 3.0

# Classical MDS: substitute for cross-component (otherwise infinite) shortest-path distances.
INFINITE_DIST_FACTOR = 1.75

# Spectral embedding: uniform background weight so disconnected nodes don't collapse
# to numerically-identical points (empirically required, not optional polish).
SPECTRAL_EPSILON = 1e-3

# Deterministic fallback-shell placement (used by kk3d/mds/spectral/node2vec for
# every node outside the giant component) and isolated-node ring placement (used
# by fa2*/fr3d/fa23d for degree-0 nodes only).
SHELL_RADIUS_FACTOR = 1.35
LOCAL_CLUSTER_RADIUS = 0.35
ISOLATED_RING_FACTOR = 1.15

# Concentric shell layout (shell_1/shell_2).
SHELL_R_BASE = 1.0
SHELL_MIN_GAP_FACTOR = 1.5

# node2vec + UMAP / PHATE / PaCMAP.
N2V_DIMENSIONS = 64
N2V_WALK_LENGTH = 30
N2V_NUM_WALKS = 200
# n_neighbors=15, min_dist=0.5, spread=2.0 found via evaluate_umap_params.py grid
# search on this data: trustworthiness stays ~flat (0.96-0.99) across the whole
# grid, while nearest-neighbor-distance CV (a direct "tight groups with big gaps"
# clumpiness proxy) roughly halves going from min_dist=0.1 to 0.5 -- i.e. the old
# min_dist=0.15 default was needlessly clumpy for near-zero fidelity gain.
N2V_UMAP_NEIGHBORS = 15
N2V_UMAP_MIN_DIST = 0.5
N2V_UMAP_SPREAD = 2.0

# Hyperbolic-radius layout (hyp_1): r = 1 - tanh(degree / HYP_BETA).
HYP_RADIUS_SCALE = 50.0

# Hive-plot-inspired 3-axis layout (hive3_1).
HIVE_AXIS_LENGTH = 50.0

RESERVED_SUFFIXES = ("_c_kv", "_kv", "_a", "_c", "_n", "_norm")


def validate_base_name(base_name: str) -> None:
    for suffix in RESERVED_SUFFIXES:
        if base_name.endswith(suffix):
            raise ValueError(
                f"layout base name {base_name!r} ends with reserved suffix {suffix!r}; "
                "the frontend uses that suffix to infer a non-layout attribute type "
                "(see src/data/ColumnTypes.ts), so a layout column must not use it."
            )


# --------------------------------------------------------------------------
# TSV IO (CRLF-preserving, no pandas, matching utils/*.py convention)
# --------------------------------------------------------------------------


def read_nodes_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return header, rows


def read_edges_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def get_existing_layout_bases(header: Iterable[str]) -> set[str]:
    """Mirrors src/data/LayoutDetection.ts getAvailableLayoutNamesFromAttributeKeys:
    a base name is available only if x_/y_/z_ all three exist."""
    keys = set(header)
    bases = set()
    for key in keys:
        if not key.startswith("x_"):
            continue
        base = key[2:]
        if base and f"y_{base}" in keys and f"z_{base}" in keys:
            bases.add(base)
    return bases


def write_layout_columns(
    nodes_path: Path,
    base_name: str,
    coords: dict[str, tuple[float, float, float]],
    *,
    precision: int = 6,
) -> bool:
    """Add or overwrite x_<base>/y_<base>/z_<base> columns in nodes.tsv.

    Re-reads the file fresh so sequential calls in one process each see prior
    runs' columns. Never reorders/touches existing columns; new columns are
    appended at the end. Writes atomically (temp file + os.replace).
    Returns True if this overwrote an existing base_name, False if new.
    """
    validate_base_name(base_name)
    header, rows = read_nodes_tsv(nodes_path)

    x_col, y_col, z_col = f"x_{base_name}", f"y_{base_name}", f"z_{base_name}"
    is_overwrite = x_col in header
    if not is_overwrite:
        header = header + [x_col, y_col, z_col]

    missing = [row["id"] for row in rows if row["id"] not in coords]
    if missing:
        raise ValueError(
            f"write_layout_columns({base_name!r}): missing coords for "
            f"{len(missing)} node ids, e.g. {missing[:5]}"
        )

    for row in rows:
        x, y, z = coords[row["id"]]
        row[x_col] = f"{float(x):.{precision}f}"
        row[y_col] = f"{float(y):.{precision}f}"
        row[z_col] = f"{float(z):.{precision}f}"

    tmp_path = nodes_path.with_name(nodes_path.name + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t", lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, nodes_path)
    return is_overwrite


def append_log_entry(
    log_path: Path,
    *,
    base_name: str,
    method: str,
    library_call: str,
    params: dict,
    weighting_desc: str,
    node_count: int,
    edge_count: int,
    fallback_notes: str = "",
    recomputed: bool = False,
    seed: int = GLOBAL_SEED,
    status: str = "OK",
) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"[{ts}] base={base_name} status={status}" + (" (recomputed)" if recomputed else ""),
        f"  method: {method}",
        f"  call: {library_call}",
        f"  params: {params}",
        f"  seed: {seed}",
        f"  weighting: {weighting_desc}",
        f"  nodes={node_count} edges={edge_count}",
    ]
    if fallback_notes:
        lines.append(f"  fallback: {fallback_notes}")
    lines.append("")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Graph building
# --------------------------------------------------------------------------


def layer3(type_a: str) -> str:
    if type_a == "metabolite":
        return "metabolite"
    if type_a == "bridge":
        return "bridge"
    return "protein"


def parse_pathways(raw: str) -> frozenset:
    """pathways_a is a comma-separated list of pathway names (empty string
    for the ~772/900 nodes with no annotation -- notably ALL metabolites,
    since pathway membership here is a gene-centric annotation)."""
    if not raw or not raw.strip():
        return frozenset()
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


def build_graph(
    node_rows: list[dict[str, str]], edge_rows: list[dict[str, str]]
) -> tuple[nx.Graph, dict[str, str]]:
    """Adds every node id from node_rows first (so degree-0 nodes are present),
    then every edge. Returns (G, node_layer). Nodes carry 'layer3' (coarse
    3-way grouping), the raw 'type_a' (5-way, incl. dual-typed
    'protein,transcript'), and 'pathways' (frozenset parsed from
    pathways_a, empty for unannotated nodes) as attributes; edges carry the
    raw 'edge_type_c' (10-way) as an attribute."""
    G = nx.Graph()
    node_layer: dict[str, str] = {}
    for row in node_rows:
        nid = row["id"]
        layer = layer3(row["type_a"])
        G.add_node(
            nid,
            layer3=layer,
            type_a=row["type_a"],
            pathways=parse_pathways(row.get("pathways_a", "")),
        )
        node_layer[nid] = layer
    for row in edge_rows:
        G.add_edge(row["source"], row["target"], edge_type_c=row.get("edge_type_c", ""))
    return G, node_layer


# --------------------------------------------------------------------------
# Edge weighting: two DIFFERENT semantics, kept as two functions on purpose.
# --------------------------------------------------------------------------


def set_attraction_weight(
    G: nx.Graph,
    node_layer: dict[str, str],
    *,
    within: float = ATTRACTION_WITHIN,
    cross: float = ATTRACTION_CROSS,
    attr: str = "fa_weight",
) -> None:
    """FA2/spring semantics: weight = attraction strength. Lower cross weight
    spreads layers apart."""
    for u, v, data in G.edges(data=True):
        data[attr] = within if node_layer[u] == node_layer[v] else cross


def set_distance_weight(
    G: nx.Graph,
    node_layer: dict[str, str],
    *,
    within: float = DISTANCE_WITHIN,
    cross: float = DISTANCE_CROSS,
    attr: str = "kk_weight",
) -> None:
    """Kamada-Kawai/MDS-hop semantics: weight = target distance. Higher cross
    weight spreads layers apart."""
    for u, v, data in G.edges(data=True):
        data[attr] = within if node_layer[u] == node_layer[v] else cross


# --------------------------------------------------------------------------
# Connectivity helpers
# --------------------------------------------------------------------------


def sorted_components(H: nx.Graph) -> list[set]:
    """Connected components of H, largest first, then min(int(id)) ascending."""
    return sorted(
        nx.connected_components(H), key=lambda c: (-len(c), min(int(n) for n in c))
    )


def giant_component_nodes(H: nx.Graph) -> set:
    return sorted_components(H)[0]


def isolated_nodes(H: nx.Graph) -> list[str]:
    """Degree-0 node ids in H, sorted by int(id)."""
    return sorted((n for n in H.nodes() if H.degree(n) == 0), key=lambda n: int(n))


# --------------------------------------------------------------------------
# Deterministic placement geometry (no RNG, reproducible across runs)
# --------------------------------------------------------------------------


def fibonacci_sphere(n: int) -> np.ndarray:
    """(n, 3) unit vectors on a sphere via a golden-angle spiral."""
    if n <= 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])
    indices = np.arange(n)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1 - (2 * indices) / (n - 1)
    radius = np.sqrt(np.clip(1 - z * z, 0, None))
    theta = golden_angle * indices
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def ring_positions_2d(n: int, radius: float, start_angle: float = 0.0) -> np.ndarray:
    """(n, 2) points evenly spaced on a circle."""
    if n <= 0:
        return np.zeros((0, 2))
    angles = start_angle + 2 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def _bounding_centroid_radius(positions: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    arr = np.array(list(positions.values()), dtype=float)
    centroid = arr.mean(axis=0)
    radius = float(np.linalg.norm(arr - centroid, axis=1).max())
    return centroid, radius


def local_component_sublayout(H: nx.Graph, *, seed: int = GLOBAL_SEED) -> dict[str, np.ndarray]:
    """3D layout for a small (2-10 node) component, centered at the origin,
    NOT yet scaled/translated to its final shell slot."""
    n = H.number_of_nodes()
    if n <= 8:
        flat = nx.circular_layout(H, dim=2)
        return {node: np.array([p[0], p[1], 0.0]) for node, p in flat.items()}
    pos = nx.spring_layout(H, dim=3, seed=seed)
    return {node: np.asarray(p) for node, p in pos.items()}


def deterministic_fallback_shell(
    connected_pos: dict[str, np.ndarray],
    remaining_nodes: Iterable[str],
    G: nx.Graph,
    *,
    shell_radius_factor: float = SHELL_RADIUS_FACTOR,
    local_cluster_radius: float = LOCAL_CLUSTER_RADIUS,
) -> dict[str, np.ndarray]:
    """Places every node in `remaining_nodes` (i.e. not already in
    `connected_pos`) on a Fibonacci-sphere shell around the bounding sphere of
    `connected_pos`. `remaining_nodes` is split into its own connected
    components (via G's induced subgraph); each component -- singleton or
    larger -- gets one shell direction. Components of size >= 2 get a small
    internal local layout scaled/translated into that slot.
    """
    remaining = set(remaining_nodes)
    if not remaining:
        return {}
    centroid, radius = _bounding_centroid_radius(connected_pos)
    sub = G.subgraph(remaining)
    comps = sorted_components(sub) if sub.number_of_nodes() else []
    # sorted_components requires every node have >=0 degree within sub; isolated
    # nodes in `sub` are their own singleton components, which is what we want.
    directions = fibonacci_sphere(len(comps))

    result: dict[str, np.ndarray] = {}
    for comp, direction in zip(comps, directions):
        comp_nodes = sorted(comp, key=lambda n: int(n))
        slot = centroid + shell_radius_factor * radius * direction
        if len(comp_nodes) == 1:
            result[comp_nodes[0]] = slot
            continue
        local = local_component_sublayout(sub.subgraph(comp_nodes))
        scale = local_cluster_radius * radius * math.sqrt(len(comp_nodes)) / math.sqrt(10)
        for node, p in local.items():
            result[node] = slot + p * scale
    return result


def place_isolated_ring_per_layer(
    xy: dict[str, np.ndarray],
    node_layer: dict[str, str],
    isolated_ids: Iterable[str],
    z_by_layer: dict[str, float],
    *,
    radius_factor: float = ISOLATED_RING_FACTOR,
) -> dict[str, np.ndarray]:
    """For z-stacked layouts (fa2_1/fa2_2/fa2_3/fa2sep_1): places degree-0
    nodes on a 2D ring around `xy`'s bounding circle, one ring per layer3
    group, at that layer's fixed z."""
    isolated_ids = list(isolated_ids)
    if not isolated_ids:
        return {}
    arr = np.array(list(xy.values()), dtype=float)
    centroid = arr.mean(axis=0)
    ring_r = float(np.linalg.norm(arr - centroid, axis=1).max()) * radius_factor

    by_layer: dict[str, list[str]] = defaultdict(list)
    for nid in isolated_ids:
        by_layer[node_layer[nid]].append(nid)

    result: dict[str, np.ndarray] = {}
    for layer, ids in by_layer.items():
        ids_sorted = sorted(ids, key=lambda n: int(n))
        pts = ring_positions_2d(len(ids_sorted), ring_r)
        z = z_by_layer[layer]
        for nid, p in zip(ids_sorted, pts):
            result[nid] = np.array([centroid[0] + p[0], centroid[1] + p[1], z])
    return result


def place_isolated_sphere_shell(
    pos3d: dict[str, np.ndarray],
    isolated_ids: Iterable[str],
    *,
    radius_factor: float = ISOLATED_RING_FACTOR,
) -> dict[str, np.ndarray]:
    """For free-3D layouts (fa23d/fr3d): places degree-0 nodes on a
    Fibonacci-sphere shell around `pos3d`'s bounding sphere."""
    isolated_ids = sorted(isolated_ids, key=lambda n: int(n))
    if not isolated_ids:
        return {}
    centroid, radius = _bounding_centroid_radius(pos3d)
    dirs = fibonacci_sphere(len(isolated_ids))
    return {
        nid: centroid + radius * radius_factor * d for nid, d in zip(isolated_ids, dirs)
    }


def spectral_positions(
    H: nx.Graph,
    n_components: int,
    *,
    weighted: bool = False,
    node_layer: dict[str, str] | None = None,
    epsilon: float = SPECTRAL_EPSILON,
) -> dict[str, np.ndarray]:
    """Spectral embedding (Laplacian eigenmaps) of H's adjacency matrix into
    `n_components` dims. A uniform epsilon background is added so
    disconnected pieces of H don't collapse to numerically-identical points
    (see method_distance_embedding.py's spectral_1 for why this matters).
    `weighted=True` requires `node_layer` and down-weights cross-layer3
    edges via ATTRACTION_WITHIN/ATTRACTION_CROSS.
    """
    from sklearn.manifold import SpectralEmbedding

    nodes = sorted(H.nodes(), key=lambda n: int(n))
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n))
    for u, v in H.edges():
        w = 1.0
        if weighted:
            w = ATTRACTION_WITHIN if node_layer[u] == node_layer[v] else ATTRACTION_CROSS
        adj[idx[u], idx[v]] = w
        adj[idx[v], idx[u]] = w
    adj += epsilon
    np.fill_diagonal(adj, 0.0)

    model = SpectralEmbedding(n_components=n_components, affinity="precomputed", random_state=GLOBAL_SEED)
    emb = model.fit_transform(adj)
    return {n: emb[idx[n]] for n in nodes}


def layer_z_stack(
    xy: dict[str, np.ndarray],
    node_layer: dict[str, str],
    *,
    fraction: float = Z_STACK_FRACTION,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Combines a 2D layout with a fixed z offset per layer3. Returns
    (coords, z_by_layer) so callers can reuse z_by_layer for isolated-node
    ring placement."""
    arr = np.array(list(xy.values()), dtype=float)
    span = max(
        float(arr[:, 0].max() - arr[:, 0].min()),
        float(arr[:, 1].max() - arr[:, 1].min()),
    )
    d = fraction * span
    z_by_layer = {"protein": -d, "bridge": 0.0, "metabolite": d}
    coords = {
        n: np.array([p[0], p[1], z_by_layer[node_layer[n]]]) for n, p in xy.items()
    }
    return coords, z_by_layer


class elapsed_spinner:
    """Per-layout progress indicator for methods with no native progress hooks
    (FA2, spring, Kamada-Kawai, spectral, shell, centrality): shows a small
    nested progress bar ticking elapsed time while the wrapped call runs.
    Methods that DO have native hooks (node2vec walk generation, UMAP,
    sklearn MDS verbose output) skip this and print their own real progress
    instead -- see run_layouts.py's `native_progress` registry flag.
    """

    def __init__(self, desc: str, position: int = 1):
        self._pbar = tqdm(
            total=0,
            desc=desc,
            bar_format="  {desc}: running... {elapsed}",
            position=position,
            leave=False,
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._tick, daemon=True)

    def _tick(self) -> None:
        while not self._stop.wait(0.2):
            self._pbar.refresh()

    def __enter__(self) -> "elapsed_spinner":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._pbar.close()


@dataclass
class GraphData:
    G: nx.Graph
    node_layer: dict[str, str]
    node_rows: list[dict[str, str]]
    header: list[str]


def load_graph(nodes_path: Path, edges_path: Path) -> GraphData:
    header, node_rows = read_nodes_tsv(nodes_path)
    edge_rows = read_edges_tsv(edges_path)
    G, node_layer = build_graph(node_rows, edge_rows)
    return GraphData(G=G, node_layer=node_layer, node_rows=node_rows, header=header)
