# AGENTS.md — ppicml layout generator

Machine-facing reference for making changes to this directory. For a
human-facing description of what each layout *does*, see `README.md` — this
file is about *how the code is built* so you can extend it correctly.

## Scope

This directory computes 3D node layouts for `public/data/ppicml/nodes.tsv`
+ `edges.tsv` (a small protein-metabolite biological network) and writes
them back into `nodes.tsv` as column triplets. It does not touch the
frontend app; it only produces data the frontend already knows how to
consume (see "Frontend contract" below). Entry point: `run_layouts.py`.

## Data model — verify before trusting these numbers

Re-derive with `python3 run_layouts.py --dry-run` (prints node/edge/
component counts) rather than hardcoding assumptions — this data can and
does change (columns have been added mid-project before, e.g. `pathways_a`
and the GO/KEGG/orphanet annotation columns arrived after the original 8).

As of this writing: **900 nodes, 1689 edges, 118 connected components**
(one giant component of 745 nodes, 14 small components of 2-10 nodes, 103
fully isolated/degree-0 nodes).

`nodes.tsv` non-layout columns (15): `id, name, degree_n, type_a,
type_simple_c, protein_id_c, transcript_id_c, metabolite_id_c, GO_BP_a,
GO_CC_a, GO_MF_a, Phenotype_Ontology_a, KEGG_a, orphanet_a, pathways_a`.

- `type_a` (5-way, use for fine-grained grouping): `protein`, `transcript`,
  `protein,transcript` (21 dual-typed nodes), `bridge`, `metabolite`.
- `layer3` (3-way coarsening, computed by `common.layer3()`, NOT a real
  column): `protein` (folds in transcript + dual-typed), `bridge`,
  `metabolite`. Most existing layouts use `layer3`; a few (`hive5_1`,
  `domaintouch_1`) use the finer `type_a` on purpose.
- `_a`-suffixed columns are comma-separated arrays (frontend convention,
  see `src/data/ColumnTypes.ts`). Only `pathways_a` is currently wired into
  any layout (`method_pathway.py`) — 428/900 nodes annotated, **zero
  metabolites** (gene-centric annotation). `GO_BP_a`/`KEGG_a`/etc. are
  present in the data but unused; the same TF-IDF + neighbor-propagation
  pattern in `method_pathway.py` would extend to them if asked.
- `degree_n` is a precomputed column but **nothing in this codebase reads
  it** — every layout uses `G.degree(n)` from the live graph built by
  `common.build_graph()`. If `nodes.tsv`'s `degree_n` and the live graph
  ever disagree, the live graph wins everywhere.

`edges.tsv` columns: `source, target, edge_type_c, type_simple_c,
edge_type_agg_c, test` (`test` is an unused legacy column). Only
`edge_type_c` (10 distinct values, e.g. `protein-bridge`,
`metabolite-metabolite`) is loaded onto the graph.

Both files are **CRLF-terminated, tab-separated, no BOM**. Preserve this —
see `common.write_layout_columns`.

## Architecture

```
common.py              shared IO, graph building, weighting, placement geometry, logging
method_forceatlas.py    fa2_*, fa2sep_*, fa23d_*, fa2spectral_1, fa2rarity_1, fa2pathway_1(*)
method_classic_force.py fr3d_*, kk3d_*
method_distance_embedding.py  mds_*, spectral_*, isomap_1
method_node2vec.py      n2vbal_1/n2vbfs_1/n2vdfs_1 + PHATE/PaCMAP/DensMAP/openTSNE variants
method_shell.py         shell_1, shell_2, shell_rarity_1
method_centrality.py    topodeg_1, topobet_1, domaintouch_1, raritytouch_1
method_hyperbolic.py    hyp_1
method_hive.py          hive3_1, hive5_1
method_community.py     community_1, community_shellz_1
method_layered_n2v.py   n2vlayered_1
method_landscape.py     landscape_1, rwr_1
method_paga.py          paga_1
method_metapath.py      metapath_1
method_ensemble.py      ensemble_1
method_pathway.py       fa2pathway_1(*), pathway_landscape_1
run_layouts.py          CLI + REGISTRY (the only place layouts are wired up)
evaluate_umap_params.py standalone grid-search tool, not part of the run pipeline
layout_run_log.txt      generated (in public/data/ppicml/, not this dir) — append-only log
```
(*) `fa2pathway_1` is registered under `method_pathway.py`, not
`method_forceatlas.py`, despite the name — pathway-related layouts live
together regardless of which base algorithm they use.

Modules are grouped by *mechanism*, not 1:1 with individual layouts — e.g.
`method_forceatlas.py` holds every direct `nx.forceatlas2_layout` caller
because they share the isolated-node-exclusion pattern; `method_centrality.py`
holds every "reuse fa2_1's xy, vary z" layout because they share
`_centrality_elevation()`. When adding a layout, check if an existing
module's shared helper already does 90% of what you need before writing a
new file.

## The layout function contract

Every layout is a function with this exact signature, referenced from a
`RunSpec` in `run_layouts.py`'s `REGISTRY`:

```python
def run_<base_name>(G: nx.Graph, node_layer: dict[str, str], nodes_path: Path
                     ) -> tuple[dict[str, tuple[float, float, float]], dict]:
    ...
    return coords, meta
```

- `G` — the full graph from `common.build_graph()`: every node from
  `nodes.tsv` present (including degree-0 ones), node attrs `layer3`,
  `type_a`, `pathways` (frozenset); edge attr `edge_type_c`.
- `node_layer` — `{node_id: layer3}`, same as `nx.get_node_attributes(G,
  'layer3')`, passed separately for convenience/history.
- `nodes_path` — only used by layouts that read back already-computed
  columns from the live file (e.g. `method_centrality.py`'s
  `_get_or_compute_fa2_xy`, which reuses `fa2_1`'s xy if present).
- Returns `coords` (must have an entry for **every** node in `G`, values
  are anything unpackable as 3 floats — tuple, list, or `np.ndarray` all
  work since `write_layout_columns` does `x, y, z = coords[id]`) and `meta`
  — a dict with keys `method`, `library_call`, `params`, `weighting_desc`,
  and optionally `fallback_notes` — all folded into the human-readable
  `layout_run_log.txt` entry via `common.append_log_entry`.

`run_layouts.py`'s main loop calls `write_layout_columns` (adds/overwrites
`x_<base>/y_<base>/z_<base>`, never touches other columns) then
`append_log_entry`, immediately after each layout — not batched — so a
crash partway through a full run still leaves prior layouts persisted.

## Disconnected-graph handling — pick the right pattern

The graph has 155 nodes outside the giant component (103 isolated + 52 in
small components). Every layout must place them somehow; there are two
established patterns — **don't invent a third without a reason**:

**Pattern A ("fa2 family")** — used when the base algorithm tolerates
disconnected input natively (ForceAtlas2, spring layout do):
1. `iso = c.isolated_nodes(G)`; exclude only those, keep small components
   in the simulation: `Gc = G.subgraph([n for n in G.nodes() if n not in iso])`.
2. Run the algorithm on `Gc`.
3. Place isolated nodes afterward: `c.place_isolated_ring_per_layer(...)`
   (for z-stacked 2D+z layouts) or `c.place_isolated_sphere_shell(...)`
   (for native-3D layouts).

**Pattern B ("giant-component family")** — used when the algorithm needs
global distance/embedding structure that's meaningless across components
(Kamada-Kawai, MDS, spectral embedding, node2vec-based methods, since walks
can't cross components):
1. `giant = c.giant_component_nodes(G)`; run the algorithm on
   `G.subgraph(giant)` only.
2. `remaining = [n for n in G.nodes() if n not in giant]`.
3. `c.deterministic_fallback_shell(giant_pos, remaining, G)` places every
   remaining node (both isolated singles AND small components) on a
   Fibonacci-sphere shell around the giant component's bounding sphere,
   with small components getting a scaled-down internal sublayout
   (`c.local_component_sublayout`) rather than a single point.

Feature-space embeddings (`landscape_1`, `rwr_1`, `pathway_landscape_1`)
use a **third, valid variant**: they don't depend on graph connectivity at
all for nodes with real feature signal, so they embed every node with a
non-degenerate feature vector directly, and only fall back to
`deterministic_fallback_shell` for nodes with *no* signal at all (e.g.
`pathway_landscape_1` propagates pathway profiles 1-2 hops before falling
back — see that file for the pattern if extending it to `GO_BP_a`/`KEGG_a`).

Never skip disconnected-node handling "because it's a small effect" — every
layout in this repo has one, and the frontend renders all 900 nodes
regardless.

## `common.py` helper reference

| Helper | Use for |
|---|---|
| `layer3(type_a)` / `build_graph(...)` / `load_graph(...)` | Graph construction (call once via `load_graph`, not per-layout) |
| `set_attraction_weight(G, node_layer, ...)` | FA2/spring edge weight: **higher = closer**. Writes to `G` edge attr `fa_weight` by default |
| `set_distance_weight(G, node_layer, ...)` | KK/MDS-hop edge weight: **higher = farther**. Opposite semantics from above — don't mix them up |
| `sorted_components(H)` / `giant_component_nodes(H)` / `isolated_nodes(H)` | Connectivity queries, deterministic ordering (size desc, then min id) |
| `fibonacci_sphere(n)` | Deterministic, RNG-free unit directions on a sphere — reuse this instead of `np.random` for any "spread N things evenly in 3D" need |
| `ring_positions_2d(n, radius)` | Same idea but a 2D circle (for z-stacked layouts) |
| `local_component_sublayout(H)` | Small-component (2-10 node) internal layout, centered at origin, not yet placed |
| `deterministic_fallback_shell(pos, remaining, G)` | Pattern B's placement step (see above) |
| `place_isolated_ring_per_layer(...)` / `place_isolated_sphere_shell(...)` | Pattern A's placement step (see above) |
| `spectral_positions(H, n_components, ...)` | Shared spectral embedding used by both `spectral_1/2` and `fa2spectral_1`'s warm-start seed |
| `layer_z_stack(xy, node_layer)` | Turns a 2D layout into z-stacked 3D (protein=-D, bridge=0, metabolite=+D, D from that run's own xy spread) — returns `z_by_layer` too, for `place_isolated_ring_per_layer` |
| `elapsed_spinner(desc)` | Context manager: nested tqdm bar showing elapsed time for a layout with no native progress hooks — used by default for every layout now (see "Progress bar" below) |
| `read_nodes_tsv` / `read_edges_tsv` / `write_layout_columns` / `append_log_entry` | TSV IO — always CRLF, always atomic write (temp file + `os.replace`) |
| `validate_base_name(name)` | Called automatically by `write_layout_columns` and `run_layouts.py`'s registry validation — raises if `name` ends in a reserved suffix |

Constants live at the top of `common.py`, each with a comment explaining
*why* that value (several were tuned empirically — e.g.
`N2V_UMAP_MIN_DIST`/`N2V_UMAP_SPREAD` came from `evaluate_umap_params.py`'s
grid search, not a guess). Change them there, not by hardcoding a
different value in a method file.

## Frontend contract (don't break this)

The viewer app (elsewhere in this repo, `src/data/LayoutDetection.ts`)
auto-detects a layout purely from the existence of all three of
`x_<name>`, `y_<name>`, `z_<name>` in `nodes.tsv`'s header — no manifest,
no registration on the frontend side. Consequences:

- `<name>` (the `base_name`) must not end in `_c`, `_n`, `_a`, `_kv`,
  `_c_kv`, or `_norm` — those suffixes make the frontend infer a
  categorical/numeric/array attribute type instead of a layout
  (`src/data/ColumnTypes.ts`). `common.validate_base_name` enforces this;
  `run_layouts.py` validates the whole registry before running anything.
- Naming convention for repeat runs of the same method with different
  params: `<tag>_<n>`, e.g. `fa2_1`, `fa2_2`, `fa2_3` (matches the
  frontend's own `_2`/`_3` collision-suffix convention in
  `src/windows/LayoutCalcWindow.ts`).
- The original non-layout columns must never be reordered or removed —
  `write_layout_columns` only appends new `x_/y_/z_` columns at the end and
  never touches anything else.

## Adding a new layout

1. Pick an existing `method_*.py` if your layout shares a mechanism with
   what's there (see the module list above); otherwise create a new
   `method_<name>.py` following the existing docstring style (explain the
   technique, cite a source if it's from a paper/tool, state which
   disconnected-graph pattern it uses and why).
2. Write `run_<base_name>(G, node_layer, nodes_path)` per the contract
   above. Reuse `common.py` helpers rather than reimplementing placement
   geometry — grep the table above first.
3. Add one `RunSpec("<base_name>", module.run_<base_name>, "<one-line
   description>")` to `run_layouts.py`'s `REGISTRY` list. Cheap/deterministic
   layouts go earlier in the list (so a partial/interrupted run still
   yields something), expensive ones (node2vec-based, ~40-120s each) later.
4. Smoke-test in isolation first: `python3 run_layouts.py --only
   <base_name> --no-progress` (see "Testing" below for the full checklist).
5. Update `README.md`'s method table (keep it alphabetically sorted by
   `base_name`) and the file list at the bottom.

## Testing / verification checklist

Run after adding or changing any layout:

```bash
python3 run_layouts.py --dry-run                    # graph loads, registry validates
python3 run_layouts.py --only <base_name> --no-progress   # isolated smoke test
```

Then verify the write, e.g.:

```python
import sys; sys.path.insert(0, "utils/ppicml_layouts")
import common as c
header, rows = c.read_nodes_tsv("public/data/ppicml/nodes.tsv")
assert len(rows) == 900                                   # row count unchanged
assert not (len(header) != len(set(header)))               # no duplicate columns
bases = c.get_existing_layout_bases(header)
assert "<base_name>" in bases                               # frontend would detect it
bad = [(r["id"], col) for r in rows for col in header
       if col.startswith(("x_", "y_", "z_")) and not math.isfinite(float(r[col]))]
assert not bad                                              # no NaN/Inf
```

Also worth a spot-check for genuinely new signals: confirm the column
isn't degenerate (e.g. `domaintouch_1`'s z has exactly 4 distinct values by
design — check the actual `nunique` matches what the metric *should*
produce, not just "some values").

`file public/data/ppicml/nodes.tsv` should still report **CRLF line
terminators**, no BOM, after any write.

## Progress bar architecture

`run_layouts.py` shows a persistent tqdm bar (`position=0`, tracks overall
`i/N`) plus a per-layout `elapsed_spinner` (`position=1`) for every layout.
This depends on **no library printing raw (non-tqdm-aware) output**:
`node2vec`'s `Node2Vec(..., quiet=True)` and every `umap.UMAP(...,
verbose=False)` call must stay that way — flipping them back to
"informative" mode will bury the bar again (confirmed painful in practice:
UMAP's `verbose=True` dumps per-epoch timestamped lines, node2vec's
`quiet=False` prints its own tqdm bars that fight with ours for the
terminal). Two harmless one-time library warnings (sklearn's
`force_all_finite` rename notice, UMAP's expected `n_jobs overridden`
notice) are explicitly filtered at the top of `run_layouts.py`. One
irreducible noise source remains: gensim/Cython prints a handful of
`Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'` lines
per node2vec run — a known interpreter-cleanup cosmetic artifact that
bypasses normal stdout/stderr routing (`sys.unraisablehook`), harmless, not
worth suppressing.

The `RunSpec.native_progress` field still exists but nothing sets it
`True` currently — it was originally used to skip the spinner for methods
with real native progress, but that traded a working persistent bar for
noisy output, so it was reverted. Leave it `False` for new layouts too.

## Dependencies and installing new ones — read this before `pip install`

This runs against the **machine's global pyenv interpreter**, shared with
many unrelated projects (confirmed: langchain, transformers, streamlit,
tooluniverse, depthai, gradio, etc. all installed here) — there is no
project-local venv for this directory. **A `pip install` here already once
silently upgraded numpy and broke several unrelated packages** (node2vec,
langchain, transformers, depthai-sdk all pin `numpy<2.0`; installing
`umap-learn` pulled `numpy>=2.x` transitively). If you need a new package:

1. Check `python3 -c "import numpy; print(numpy.__version__)"` before.
2. `pip install <package>` — watch for `ERROR: pip's dependency resolver
   ...` conflict warnings in the output, not just a clean exit code.
3. Re-check numpy's version and re-import the critical existing packages
   (`node2vec`, `networkx`, `umap`, `sklearn`) after.
4. If numpy got bumped and something broke: `pip install "numpy==<old
   version>"` to pin it back — this is what fixed it last time.

Currently-added packages beyond a bare `networkx`/`scikit-learn`/`scipy`/
`numpy` stack: `umap-learn`, `node2vec`, `gensim`, `python-louvain`,
`tqdm`, `phate`, `pacmap`, `openTSNE` — all installed cleanly with no
conflicts (checked each time). See `README.md`'s Dependencies section for
the full `pip install` line.

## Other gotchas learned the hard way

- **Don't run two node2vec/numba-heavy layouts concurrently** (e.g. two
  background `run_layouts.py --only ...` invocations at once, or a manual
  test script alongside a batch run). Observed once: a PaCMAP run died
  silently (exit 0, no traceback, no output past the walk-generation step)
  while a UMAP grid-search was running concurrently in another process —
  root cause was resource contention (numba JIT compilation is CPU/memory
  heavy), not a real bug; re-running alone succeeded immediately. If a
  layout appears to hang or die with no error, check `ps`/`uptime` for
  concurrent heavy processes before assuming the code is broken.
- **`openTSNE`'s `negative_gradient_method`** valid values are `"auto"`,
  `"bh"` (Barnes-Hut, works fine for `n_components=3`), or `"fft"`
  (interpolation-based, **2D-only**) — `"exact"` is not a real option in
  this version and raises `ValueError`. Use `"bh"` explicitly for 3D.
- **`node2vec`'s `quiet` flag only toggles its internal tqdm bars** — it
  does not change `workers`/parallelism, so don't expect (or debug for) a
  speed difference between `quiet=True`/`False`. Run-to-run timing
  variance of 1.5-2x on this shared desktop machine is normal system load,
  not a regression.
- **High-cardinality categorical annotation data** (e.g. `pathways_a`'s
  1338 distinct terms across 428 nodes) needs TF-IDF weighting +
  dimensionality reduction (`TruncatedSVD`) before a final UMAP/PaCMAP
  step — raw one-hot/multi-hot straight into UMAP is noisy and slow at
  this cardinality. See `method_pathway.py`'s `_build_pathway_matrix` for
  the established pattern (also handles missing-annotation nodes via 1-2
  hop neighbor mean-pooling before the TF-IDF step).
