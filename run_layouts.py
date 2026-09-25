#!/usr/bin/env python3
"""Compute a battery of 3D graph layouts for the ppicml protein/metabolite
multilayer network and bake them into nodes.tsv as x_<name>/y_<name>/z_<name>
column triplets (the naming convention the viewer app auto-detects, see
src/data/LayoutDetection.ts).

Usage:
    python run_layouts.py                       # run every layout in the registry
    python run_layouts.py --only fa2_1,mds_1     # run a subset
    python run_layouts.py --list                 # print the registry, no writes
    python run_layouts.py --dry-run              # build the graph, validate names, no writes
    python run_layouts.py --nodes PATH --edges PATH --log PATH   # override default paths

Every run appends a human-readable entry to layout_run_log.txt (same
directory as nodes.tsv) recording its parameters, weighting scheme, and any
disconnected-graph fallback notes.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

# Two known-benign, one-time library warnings that would otherwise print
# straight to stderr (not tqdm-aware) and interrupt the persistent progress
# bar: sklearn's internal `force_all_finite` rename notice, and UMAP's
# expected "n_jobs overridden" notice (fires because we intentionally pass
# random_state for reproducibility -- that's the point, not a mistake).
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*", category=UserWarning)

import common as c
import method_centrality as m_centrality
import method_classic_force as m_classic
import method_community as m_community
import method_distance_embedding as m_dist
import method_ensemble as m_ensemble
import method_forceatlas as m_fa2
import method_hive as m_hive
import method_hyperbolic as m_hyp
import method_landscape as m_landscape
import method_layered_n2v as m_layered
import method_metapath as m_metapath
import method_node2vec as m_n2v
import method_paga as m_paga
import method_pathway as m_pathway
import method_shell as m_shell

DEFAULT_DATA_DIR = (Path(__file__).resolve().parent.parent.parent / "public" / "data" / "ppicml")


@dataclass
class RunSpec:
    base_name: str
    func: Callable
    description: str
    # Reserved: True would skip the elapsed-time spinner for a method that prints
    # its own real, tqdm-aware progress. Not used by anything currently -- node2vec
    # (quiet=True) and UMAP/openTSNE/PHATE/PaCMAP (verbose=False) are all silenced
    # everywhere so the persistent global bar stays visible at the bottom of the
    # terminal instead of being buried by raw (non-tqdm-aware) library output.
    native_progress: bool = False


REGISTRY: list[RunSpec] = [
    # Deterministic / cheap first, so a partial run still yields useful results fast.
    RunSpec("shell_1", m_shell.run_shell_1, "Concentric shells: protein/bridge/metabolite"),
    RunSpec("shell_2", m_shell.run_shell_2, "Concentric shells: reversed radius order"),
    RunSpec("fa2_1", m_fa2.run_fa2_1, "ForceAtlas2, shared xy, discrete z per layer3"),
    RunSpec("fa2_2", m_fa2.run_fa2_2, "ForceAtlas2, linlog+dissuade_hubs, z-stacked"),
    RunSpec("fa2_3", m_fa2.run_fa2_3, "ForceAtlas2, cross-layer down-weighted, z-stacked"),
    RunSpec("fa2sep_1", m_fa2.run_fa2sep_1, "ForceAtlas2 run independently per layer, z-stacked"),
    RunSpec("fa23d_1", m_fa2.run_fa23d_1, "ForceAtlas2, native 3D"),
    RunSpec("fa23d_2", m_fa2.run_fa23d_2, "ForceAtlas2, native 3D, cross-layer down-weighted"),
    RunSpec("fr3d_1", m_classic.run_fr3d_1, "Fruchterman-Reingold / spring, native 3D"),
    RunSpec("fr3d_2", m_classic.run_fr3d_2, "Fruchterman-Reingold, cross-layer down-weighted"),
    RunSpec("topodeg_1", m_centrality.run_topodeg_1, "fa2_1 xy + z from degree centrality"),
    RunSpec("topobet_1", m_centrality.run_topobet_1, "fa2_1 xy + z from betweenness centrality"),
    RunSpec("kk3d_1", m_classic.run_kk3d_1, "Kamada-Kawai on giant component + fallback shell"),
    RunSpec("kk3d_2", m_classic.run_kk3d_2, "Kamada-Kawai, cross-layer distance-inflated"),
    RunSpec("mds_1", m_dist.run_mds_1, "Classical MDS on hop-distance matrix"),
    RunSpec("mds_2", m_dist.run_mds_2, "Classical MDS, cross-layer hop length inflated"),
    RunSpec("spectral_1", m_dist.run_spectral_1, "Spectral embedding (Laplacian eigenmaps)"),
    RunSpec("spectral_2", m_dist.run_spectral_2, "Spectral embedding, cross-layer down-weighted"),
    RunSpec("n2vbal_1", m_n2v.run_n2vbal_1, "node2vec (p=1,q=1) + UMAP(3D)"),
    RunSpec("n2vbfs_1", m_n2v.run_n2vbfs_1, "node2vec (p=0.25,q=4, BFS-like) + UMAP(3D)"),
    RunSpec("n2vdfs_1", m_n2v.run_n2vdfs_1, "node2vec (p=4,q=0.25, DFS-like) + UMAP(3D)"),
    # Added after deep-research pass (multilayer/biological network viz literature): see README.md.
    RunSpec("fa2spectral_1", m_fa2.run_fa2spectral_1, "ForceAtlas2 seeded with a spectral embedding (fCoSE-style)"),
    RunSpec("isomap_1", m_dist.run_isomap_1, "Isomap on giant component's hop-distance matrix"),
    RunSpec("hyp_1", m_hyp.run_hyp_1, "Hyperbolic-radius layout (r=1-tanh(degree/beta)), layer-banded"),
    RunSpec("hive3_1", m_hive.run_hive3_1, "Hive-plot-style 3-axis layout, one spoke per layer3"),
    RunSpec("hive5_1", m_hive.run_hive5_1, "Hive-plot-style 5-axis layout, one spoke per raw type_a value"),
    RunSpec("community_1", m_community.run_community_1, "Louvain community meta-layout + local sublayout"),
    RunSpec("community_shellz_1", m_community.run_community_shellz_1, "community_1 xy + shell_1 z-formula by predominant layer"),
    RunSpec("n2vbal_phate_1", m_n2v.run_n2vbal_phate_1, "node2vec (p=1,q=1) + PHATE(3D)"),
    RunSpec("n2vbal_pacmap_1", m_n2v.run_n2vbal_pacmap_1, "node2vec (p=1,q=1) + PaCMAP(3D)"),
    RunSpec("n2vbfs_pacmap_1", m_n2v.run_n2vbfs_pacmap_1, "node2vec (p=0.25,q=4, BFS-like) + PaCMAP(3D)"),
    RunSpec("n2vdfs_pacmap_1", m_n2v.run_n2vdfs_pacmap_1, "node2vec (p=4,q=0.25, DFS-like) + PaCMAP(3D)"),
    RunSpec(
        "n2vlayered_1",
        m_layered.run_n2vlayered_1,
        "Separate protein/metabolite node2vec+UMAP slabs, z-squeezed and offset, "
        "bridge nodes placed by harmonic relaxation",
    ),
    # Added after the second (top-20-idea) deep-research pass: see README.md.
    RunSpec("landscape_1", m_landscape.run_landscape_1, "VRNetzer-style functional landscape (feature-matrix UMAP)"),
    RunSpec("rwr_1", m_landscape.run_rwr_1, "Random-walk-with-restart feature landscape (personalized PageRank + UMAP)"),
    RunSpec("paga_1", m_paga.run_paga_1, "PAGA-style two-level coarse (Louvain meta-graph) + fine (spring) layout"),
    RunSpec("metapath_1", m_metapath.run_metapath_1, "Domain-continuity-biased random walks + Word2Vec + UMAP(3D)"),
    RunSpec("ensemble_1", m_ensemble.run_ensemble_1, "Procrustes-aligned blend of fa23d_1 + mds_1 + spectral_1"),
    RunSpec("domaintouch_1", m_centrality.run_domaintouch_1, "fa2_1 xy + z from # distinct domains touched by neighborhood"),
    RunSpec("raritytouch_1", m_centrality.run_raritytouch_1, "fa2_1 xy + z from rarest incident edge_type_c"),
    RunSpec("n2vbal_densmap_1", m_n2v.run_n2vbal_densmap_1, "node2vec (p=1,q=1) + DensMAP(3D)"),
    RunSpec("fa2rarity_1", m_fa2.run_fa2rarity_1, "ForceAtlas2 with full 10-bucket edge_type_c rarity weighting"),
    RunSpec("shell_rarity_1", m_shell.run_shell_rarity_1, "Concentric shells banded by rarest incident edge_type_c"),
    RunSpec("n2vbal_tsne_1", m_n2v.run_n2vbal_tsne_1, "node2vec (p=1,q=1) + openTSNE(3D)"),
    # Added once real pathways_a annotation was added to nodes.tsv: see README.md.
    RunSpec("fa2pathway_1", m_pathway.run_fa2pathway_1, "ForceAtlas2 with real edges boosted by shared-pathway Jaccard similarity"),
    RunSpec("pathway_landscape_1", m_pathway.run_pathway_landscape_1, "Pure pathways_a functional landscape (TF-IDF+SVD+UMAP, neighbor-propagated)"),
]


def validate_registry() -> None:
    seen = set()
    for spec in REGISTRY:
        c.validate_base_name(spec.base_name)
        if spec.base_name in seen:
            raise ValueError(f"duplicate base_name in REGISTRY: {spec.base_name!r}")
        seen.add(spec.base_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default=None, help="comma-separated base_names to run")
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    parser.add_argument("--dry-run", action="store_true", help="build graph, validate, no writes")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_DATA_DIR / "nodes.tsv")
    parser.add_argument("--edges", type=Path, default=DEFAULT_DATA_DIR / "edges.tsv")
    parser.add_argument("--log", type=Path, default=DEFAULT_DATA_DIR / "layout_run_log.txt")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    args = parser.parse_args()

    validate_registry()

    if args.list:
        for spec in REGISTRY:
            print(f"{spec.base_name:12s} {spec.description}")
        return 0

    selected = REGISTRY
    if args.only:
        wanted = set(x.strip() for x in args.only.split(","))
        valid = {spec.base_name for spec in REGISTRY}
        unknown = wanted - valid
        if unknown:
            print(f"Unknown base_name(s): {sorted(unknown)}", file=sys.stderr)
            print(f"Valid names: {sorted(valid)}", file=sys.stderr)
            return 2
        selected = [spec for spec in REGISTRY if spec.base_name in wanted]

    print(f"Loading graph from {args.nodes} / {args.edges} ...")
    data = c.load_graph(args.nodes, args.edges)
    G, node_layer = data.G, data.node_layer
    comps = c.sorted_components(G)
    n_isolated = len(c.isolated_nodes(G))
    print(
        f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
        f"components={len(comps)} giant={len(comps[0])} isolated={n_isolated}"
    )

    if args.dry_run:
        print(f"Would run {len(selected)} layout(s):")
        for spec in selected:
            print(f"  {spec.base_name}")
        return 0

    existing_bases = c.get_existing_layout_bases(data.header)
    failures: list[str] = []

    show_progress = not args.no_progress
    global_bar = tqdm(
        selected,
        desc="Layouts",
        unit="layout",
        position=0,
        disable=not show_progress,
    )
    for spec in global_bar:
        global_bar.set_description(f"Layouts ({spec.base_name})")
        start = time.time()
        tqdm.write(f"-> {spec.base_name} ({spec.description}) ...")
        try:
            if spec.native_progress or not show_progress:
                coords, meta = spec.func(G, node_layer, args.nodes)
            else:
                with c.elapsed_spinner(spec.base_name):
                    coords, meta = spec.func(G, node_layer, args.nodes)
            coords_tuples = {n: tuple(float(v) for v in p) for n, p in coords.items()}
            recomputed = c.write_layout_columns(args.nodes, spec.base_name, coords_tuples)
            c.append_log_entry(
                args.log,
                base_name=spec.base_name,
                method=meta["method"],
                library_call=meta["library_call"],
                params=meta["params"],
                weighting_desc=meta["weighting_desc"],
                node_count=G.number_of_nodes(),
                edge_count=G.number_of_edges(),
                fallback_notes=meta.get("fallback_notes", ""),
                recomputed=recomputed or spec.base_name in existing_bases,
                status="OK",
            )
            existing_bases.add(spec.base_name)
            elapsed = time.time() - start
            tqdm.write(f"   done in {elapsed:.1f}s")
        except Exception as exc:  # noqa: BLE001 - batch script must continue past a single failure
            traceback.print_exc()
            c.append_log_entry(
                args.log,
                base_name=spec.base_name,
                method=spec.description,
                library_call="(failed before/during computation)",
                params={},
                weighting_desc="",
                node_count=G.number_of_nodes(),
                edge_count=G.number_of_edges(),
                recomputed=False,
                status=f"FAILED: {exc}",
            )
            failures.append(spec.base_name)
    global_bar.close()

    print()
    if failures:
        print(f"{len(failures)}/{len(selected)} layout(s) FAILED: {failures}", file=sys.stderr)
        return 1
    print(f"All {len(selected)} layout(s) completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
