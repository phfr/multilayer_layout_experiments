#!/usr/bin/env python3
"""Grid-search UMAP parameters (n_neighbors, min_dist, spread) for the
node2vec embeddings used by n2vbal_1/n2vbfs_1/n2vdfs_1/n2vlayered_1/
n2vbal_phate_1, scoring each combination on two axes:

  - trustworthiness: how faithfully the 3D embedding preserves the original
    64-dim node2vec neighborhood structure (sklearn.manifold.trustworthiness,
    0-1, higher = more faithful). A purely "prettier" embedding that ignores
    the underlying structure would score low here.
  - spread evenness: coefficient of variation (std/mean) of each point's
    nearest-neighbor distance in the 3D embedding. High CV means some points
    are crammed together while others sit in near-empty space -- exactly
    the "tight groups with big gaps" symptom being diagnosed. Lower is more
    evenly spread.

Prints a ranked table and a recommended (n_neighbors, min_dist, spread)
balancing both (normalized trust - normalized cv).

Read-only: takes the same --input-dir / --layer-order as run_layouts.py and
writes nothing.

Usage:
    python evaluate_umap_params.py -i INPUT_DIR
    python evaluate_umap_params.py -i INPUT_DIR --p 0.25 --q 4   # evaluate for n2vbfs's walk bias
    python evaluate_umap_params.py -i INPUT_DIR --n-neighbors-grid 10,15,30,50 \\
        --min-dist-grid 0.1,0.3,0.5,0.8 --spread-grid 1.0,1.5,2.0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import umap
from scipy.spatial import cKDTree
from sklearn.manifold import trustworthiness

import common as c
from method_node2vec import _node2vec_embedding


def nn_distance_cv(points: np.ndarray) -> float:
    """Coefficient of variation of nearest-neighbor distances: a cheap,
    direct proxy for visual clumpiness (high = tight groups + big gaps)."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)  # column 0 is self (dist 0)
    nn_dist = dists[:, 1]
    return float(nn_dist.std() / nn_dist.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=1.0, help="node2vec p (default: balanced n2vbal)")
    parser.add_argument("--q", type=float, default=1.0, help="node2vec q (default: balanced n2vbal)")
    parser.add_argument("--n-neighbors-grid", type=str, default="10,15,30,50")
    parser.add_argument("--min-dist-grid", type=str, default="0.1,0.3,0.5,0.8")
    parser.add_argument("--spread-grid", type=str, default="1.0,1.5,2.0")
    parser.add_argument("--trust-k", type=int, default=10, help="neighborhood size for trustworthiness")
    parser.add_argument(
        "-i", "--input-dir", type=Path, required=True,
        help=f"folder containing {c.NODES_FILENAME} and {c.EDGES_FILENAME} (read-only)",
    )
    parser.add_argument(
        "--layer-order", type=str, default=",".join(c.LAYER_ORDER),
        help=f"comma-separated {c.LAYER_COLUMN} values, as for run_layouts.py (default: %(default)s)",
    )
    args = parser.parse_args()

    print(f"Loading graph and computing node2vec embedding (p={args.p}, q={args.q}) ...", flush=True)
    data = c.load_graph(
        args.input_dir / c.NODES_FILENAME,
        args.input_dir / c.EDGES_FILENAME,
        layer_order=c.parse_layer_order(args.layer_order),
    )
    _, giant_nodes, vectors = _node2vec_embedding(data.G, p=args.p, q=args.q)
    print(f"Embedding shape: {vectors.shape}\n", flush=True)

    n_neighbors_grid = [int(x) for x in args.n_neighbors_grid.split(",")]
    min_dist_grid = [float(x) for x in args.min_dist_grid.split(",")]
    spread_grid = [float(x) for x in args.spread_grid.split(",")]

    print(f"{'n_neighbors':>11} {'min_dist':>9} {'spread':>7} {'trust':>7} {'nn_cv':>7} {'time':>6}")
    results = []
    for n_neighbors in n_neighbors_grid:
        for min_dist in min_dist_grid:
            for spread in spread_grid:
                if min_dist > spread:
                    continue  # UMAP requires min_dist <= spread
                start = time.time()
                reducer = umap.UMAP(
                    n_components=3,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    spread=spread,
                    metric="cosine",
                    random_state=c.GLOBAL_SEED,
                    verbose=False,
                )
                emb = reducer.fit_transform(vectors)
                elapsed = time.time() - start

                trust = trustworthiness(vectors, emb, n_neighbors=args.trust_k, metric="cosine")
                cv = nn_distance_cv(emb)
                results.append((n_neighbors, min_dist, spread, trust, cv))
                print(
                    f"{n_neighbors:>11d} {min_dist:>9.1f} {spread:>7.1f} "
                    f"{trust:>7.4f} {cv:>7.3f} {elapsed:>5.1f}s",
                    flush=True,
                )

    trusts = np.array([r[3] for r in results])
    cvs = np.array([r[4] for r in results])
    trust_norm = (trusts - trusts.min()) / (trusts.max() - trusts.min() + 1e-9)
    cv_norm = (cvs - cvs.min()) / (cvs.max() - cvs.min() + 1e-9)
    score = trust_norm - cv_norm  # reward faithfulness, penalize clumpiness equally

    order = np.argsort(-score)
    print("\nTop 5 by (trustworthiness - clumpiness), balanced equally:")
    for i in order[:5]:
        n_neighbors, min_dist, spread, trust, cv = results[i]
        print(
            f"  n_neighbors={n_neighbors:<3d} min_dist={min_dist:<4.1f} spread={spread:<4.1f}  "
            f"trust={trust:.4f}  nn_dist_cv={cv:.3f}  score={score[i]:.3f}"
        )

    best = results[int(order[0])]
    print(
        f"\nRecommended: n_neighbors={best[0]}, min_dist={best[1]}, spread={best[2]} "
        f"(trust={best[3]:.4f}, nn_dist_cv={best[4]:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
