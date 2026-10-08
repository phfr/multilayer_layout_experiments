# AGENTS.md — multilayer layout generator

Machine-facing reference for making changes to this directory. For a
human-facing description of what each layout *does*, see `README.md` — this
file is about *how the code is built* so you can extend it correctly.

## Scope

This directory computes 3D node layouts for a multilayer biological network
given as an **input folder** holding `nodes.tsv` + `edges.tsv`, and writes
them into an **output folder** as column triplets appended to a copy of
`nodes.tsv`. The input folder is never written to. It does not touch the
viewer app; it only produces data the frontend already knows how to consume
(see "Frontend contract" below). Entry point: `run_layouts.py`:

```bash
python3 run_layouts.py -i INPUT_DIR -o OUTPUT_DIR [--only a,b] [--layer-order t,p,m] \
    [--id-column id] [--layer-column layer_c] [--type-column type_a|none] \
    [--pathway-column pathways_a|none] [--source-column source] [--target-column target] \
    [--edge-type-column edge_type_c|none]
```

Output folder contents (see `common.prepare_output_dir`):

- `nodes.tsv` — input columns + one `x_/y_/z_` triplet per layout. Seeded
  by copying the input on first use; on a later run into the same folder it
  is rebuilt as input columns + the layout columns already there (carried
  over by node id), so `--only` runs accumulate and layouts that read
  `fa2_1` back keep working. A different node-id set is refused.
- `edges.tsv` — verbatim copy, so the folder is a complete dataset.
- `layout_run_log.txt` — append-only; every run opens with a banner whose
  first line is the exact command (`shlex.join(sys.argv)`), then interpreter,
  cwd, input/output paths, graph stats, the column-role mapping, layer order
  (flagged `[inferred: ...]` when `--layer-order` was not given), and the
  selected/skipped layouts (`common.append_run_header`), followed by one
  entry per layout.

Pointing `-o` at the input folder is allowed and degrades to in-place mode.

## Data model — column roles, not column names

Nothing in a method file reads `nodes.tsv` / `edges.tsv` through a literal
column name. `common.ColumnSpec` maps seven **roles** to input columns, the
CLI fills it from the `--*-column` flags (`common.add_column_arguments` /
`columns_from_args`, shared with `evaluate_umap_params.py`), and
`build_graph` turns the roles into fixed graph attributes that method code
reads:

| Role | `ColumnSpec` field / flag | Default column | Where it ends up |
|---|---|---|---|
| node id | `id` / `--id-column` | `id` | the node key in `G` |
| layer | `layer` / `--layer-column` | `layer_c` | node attr `layer`; `node_layer` dict; `c.layer_order(G)` |
| type (optional) | `type` / `--type-column` | `type_a` | node attr `type` (`""` if none) |
| pathways (optional) | `pathways` / `--pathway-column` | `pathways_a` | node attr `pathways` (frozenset, parsed from a comma-separated `_a` column) |
| source / target | `source`, `target` / `--source-column`, `--target-column` | `source`, `target` | edge endpoints |
| edge type (optional) | `edge_type` / `--edge-type-column` | `edge_type_c` | edge attr `edge_type` (`""` if none) |

The spec is stored on the graph (`c.columns(G)`) so a layout can name the
real column in its log text (e.g. `hive5_1`: "one spoke per raw type_a
value"), but **layout logic must read the attributes**, never the column.
Optional roles can be `None` (`--type-column none`); a configured-but-absent
optional column is tolerated (attribute empty) and only matters through
`RunSpec.requires` (below). Required roles (id, layer, source, target)
missing from the header are a load error naming the flag to pass.

Re-derive the numbers below with `python3 run_layouts.py -i DIR --dry-run`
(prints node/edge/component counts, the column mapping, per-layer counts
and which layouts would be skipped) rather than hardcoding assumptions —
the data has been swapped wholesale once already (a 900-node
protein/bridge/metabolite network with 10 edge types and pathway/GO
annotation columns → the current one), and layouts that hardcoded the old
layer names or edge-type counts all had to change. Treat layer names,
edge-type vocabularies and annotation columns as **data-driven**, never as
constants in a method file.

As of this writing (`wwiznet_ppimetabolite/prepare/raw_input`): **263
nodes, 317 edges, 1 connected component, 0 isolated nodes**.

`nodes.tsv` columns (8): `id, name, degree_n, layer_c, type_a,
protein_id_c, transcript_id_c, metabolite_id_c`.

- **`layer_c` is the layer column** (default `ColumnSpec.layer`). Values:
  `transcript` (141), `protein` (90), `metabolite` (32). The stacking order
  — bottom/innermost first — comes from `--layer-order`; when not given it
  is **inferred as descending node count** (ties by first appearance,
  `common.infer_layer_order` / `INFERRED_LAYER_ORDER_RULE`), which on this
  data is `transcript, protein, metabolite` — the same order the old
  hardcoded default had, so existing outputs are reproduced without flags.
  There is no hardcoded layer list anywhere any more. `build_graph` refuses
  an explicit order that misses a value present in the data or names one
  with no nodes, so the two always agree. The effective order is stored on
  the graph as `G.graph["layer_order"]`; **method code must read it via
  `c.layer_order(G)`**, never spell layer names out.
- `type_a` is a finer per-node type (currently one value per node, e.g.
  `protein_bridge`, `transcript_differentially_expressed`; 5 values). Loaded
  as node attr `type` (`""` if there is no type column). Used only by
  `hive5_1` (one spoke per value; `requires=("type",)`) and `landscape_1`
  (one-hot feature block, simply absent without the column).
- `pathways_a` (optional, absent in the current data) is the only annotation
  column any layout consumes. `fa2pathway_1` / `pathway_landscape_1` declare
  `requires=("pathways",)` on their `RunSpec` and are **skipped with a
  `SKIPPED` log entry** (not failed) when the role has no usable column.
  `_a`-suffixed columns are comma-separated arrays (frontend convention).
- `degree_n` is a precomputed column but **nothing in this codebase reads
  it** — every layout uses `G.degree(n)` from the live graph built by
  `common.build_graph()`. If `nodes.tsv`'s `degree_n` and the live graph
  ever disagree, the live graph wins everywhere.
- Node ids are arbitrary strings. Every stable ordering goes through
  `c.node_key` (numeric-looking ids sort numerically, others
  lexicographically after them) — **never `int(id)`**, which is what used to
  tie the code to integer ids.

`edges.tsv` columns: `source, target, edge_type_c`. `edge_type_c` (5 distinct
values: `protein-protein`, `transcript-transcript`, `metabolite-metabolite`,
`protein-metabolite`, `transcript-metabolite`) is loaded as edge attr
`edge_type`. The layouts that need it (`fa2rarity_1`, `raritytouch_1`,
`shell_rarity_1`) declare `requires=("edge_type",)` and are skipped without
it; `landscape_1` just drops its edge-type feature block; `metapath_1`
defines "same domain" from the endpoints' `layer` attributes (identical to
the `a-b` string convention on this data) and so needs no edge types at
all. Note there are **no transcript-protein edges**: both non-metabolite
layers connect to each other only through metabolites, which matters for
any layout that fixes the outer layers and relaxes the middle one
(`n2vlayered_1`). An edge whose endpoint is not a node id is an error.

Both files are **tab-separated, no BOM**. The current input is
LF-terminated (the previous dataset was CRLF); `write_layout_columns`
detects and preserves whichever the file already uses
(`common.detect_line_terminator`).

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
evaluate_umap_params.py standalone grid-search tool (-i INPUT_DIR, read-only), not part of the run pipeline
OUTPUT_DIR/layout_run_log.txt   generated — append-only log, lives next to the output nodes.tsv
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
  `nodes.tsv` present (including degree-0 ones), node attrs `layer`,
  `type`, `pathways` (frozenset); edge attr `edge_type`; graph attrs
  `layer_order` / `layer_column` / `columns` (read via `c.layer_order(G)` /
  `c.columns(G)`).
- `node_layer` — `{node_id: layer}`, same as `nx.get_node_attributes(G,
  'layer')`, passed separately for convenience/history.
- `nodes_path` — the **output** `nodes.tsv` (already seeded from the input
  before the first layout runs). Only used by layouts that read back
  already-computed columns (e.g. `method_centrality.py`'s
  `_get_or_compute_fa2_xy`, which reuses `fa2_1`'s xy if present — keyed by
  `c.columns(G).id`, not a literal `"id"`).
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

The current input is a single connected component, but the previous one
had 155 nodes outside the giant component (103 isolated + 52 in small
components) and the next one may again. Every layout must place such nodes
somehow; there are two established patterns — **don't invent a third
without a reason**:

**Pattern A ("fa2 family")** — used when the base algorithm tolerates
disconnected input natively (ForceAtlas2, spring layout do):
1. `iso = c.isolated_nodes(G)`; exclude only those, keep small components
   in the simulation: `Gc = c.ordered_subgraph(G, [n for n in G.nodes() if n not in iso])`.
2. Run the algorithm on `Gc`.
3. Place isolated nodes afterward: `c.place_isolated_ring_per_layer(...)`
   (for z-stacked 2D+z layouts) or `c.place_isolated_sphere_shell(...)`
   (for native-3D layouts).

**Pattern B ("giant-component family")** — used when the algorithm needs
global distance/embedding structure that's meaningless across components
(Kamada-Kawai, MDS, spectral embedding, node2vec-based methods, since walks
can't cross components):
1. `giant = c.giant_component_nodes(G)`; run the algorithm on
   `c.ordered_subgraph(G, giant)` only.
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
layout in this repo has one, and the frontend renders every node
regardless. Per-layer sub-layouts have the same issue one level down: a
layer's induced subgraph can be tiny or fragmented even when the whole
graph is connected (`n2vlayered_1` falls back to a spring layout for a slab
whose giant component has fewer than `MIN_NODES_FOR_UMAP` nodes).

## `common.py` helper reference

| Helper | Use for |
|---|---|
| `ColumnSpec` / `DEFAULT_COLUMNS` / `add_column_arguments(parser)` / `columns_from_args(args)` | The column-role model and its CLI flags (shared by `run_layouts.py` and `evaluate_umap_params.py`) |
| `build_graph(..., columns=, layer_order=)` / `load_graph(...)` / `parse_layer_order(s)` / `infer_layer_order(rows, col)` | Graph construction (call once via `load_graph`, not per-layout); maps roles to attrs, validates the layer column against an explicit order or infers one |
| `layer_order(G)` | The run's layer stacking order, first = bottom/innermost — **the only sanctioned way for a method to learn layer names** |
| `columns(G)` | The run's `ColumnSpec` — for naming the real input column in log text only, never for layout logic |
| `missing_role_columns(...)` / `GraphData.missing_role_columns(roles)` | Why an optional role (`type` / `pathways` / `edge_type`) is unusable on this input; drives `RunSpec.requires` skipping |
| `node_key(n)` | Sort key for node ids (numeric-aware, string-safe) — use for every deterministic node ordering instead of `int(id)` |
| `layer_axis_offsets(order, step)` | Signed, zero-centred offsets along one axis per layer (3 layers → `-step, 0, +step`) |
| `prepare_output_dir(input_dir, output_dir, id_column=)` | Seeds/rebuilds the output `nodes.tsv` (rows matched by the id column), copies `edges.tsv`, returns the output paths (`OutputPaths`) |
| `append_run_header(log, argv=, details=)` | The per-run log banner (exact command line first) |
| `detect_line_terminator(path)` | `"\r\n"` or `"\n"`, so rewrites keep the input's convention |
| `set_attraction_weight(G, node_layer, ...)` | FA2/spring edge weight: **higher = closer**. Writes to `G` edge attr `fa_weight` by default |
| `set_distance_weight(G, node_layer, ...)` | KK/MDS-hop edge weight: **higher = farther**. Opposite semantics from above — don't mix them up |
| `sorted_components(H)` / `giant_component_nodes(H)` / `isolated_nodes(H)` | Connectivity queries, deterministic ordering (size desc, then min id) |
| `ordered_subgraph(G, nodes)` | Induced subgraph as a real graph in G's node/adjacency order — **the only way to build a subgraph that a layout algorithm consumes** (see gotchas) |
| `fibonacci_sphere(n)` | Deterministic, RNG-free unit directions on a sphere — reuse this instead of `np.random` for any "spread N things evenly in 3D" need |
| `ring_positions_2d(n, radius)` | Same idea but a 2D circle (for z-stacked layouts) |
| `local_component_sublayout(H)` | Small-component (2-10 node) internal layout, centered at origin, not yet placed |
| `deterministic_fallback_shell(pos, remaining, G)` | Pattern B's placement step (see above) |
| `place_isolated_ring_per_layer(...)` / `place_isolated_sphere_shell(...)` | Pattern A's placement step (see above) |
| `spectral_positions(H, n_components, ...)` | Shared spectral embedding used by both `spectral_1/2` and `fa2spectral_1`'s warm-start seed |
| `layer_z_stack(xy, node_layer, order=c.layer_order(G))` | Turns a 2D layout into z-stacked 3D: one plane per layer in `order`, `Z_STACK_FRACTION` × that run's own xy spread apart, centred on z=0 — returns `z_by_layer` too, for `place_isolated_ring_per_layer` |
| `elapsed_spinner(desc)` | Context manager: nested tqdm bar showing elapsed time for a layout with no native progress hooks — used by default for every layout now (see "Progress bar" below) |
| `read_nodes_tsv` / `read_edges_tsv` / `write_layout_columns(..., id_column=)` / `append_log_entry` | TSV IO — line terminator preserved from the file, always atomic write (temp file + `os.replace`) |
| `validate_base_name(name)` | Called automatically by `write_layout_columns` and `run_layouts.py`'s registry validation — raises if `name` ends in a reserved suffix |

Constants live at the top of `common.py`, each with a comment explaining
*why* that value (several were tuned empirically — e.g.
`N2V_UMAP_MIN_DIST`/`N2V_UMAP_SPREAD` came from `evaluate_umap_params.py`'s
grid search, not a guess). Change them there, not by hardcoding a
different value in a method file.

## Frontend contract (don't break this)

The viewer app (a separate repo, `src/data/LayoutDetection.ts`)
auto-detects a layout purely from the existence of all three of
`x_<name>`, `y_<name>`, `z_<name>` in `nodes.tsv`'s header — no manifest,
no registration on the frontend side. Point it at the **output** folder.
Consequences:

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
  never touches anything else. `layer_c` itself is a categorical (`_c`)
  attribute on the frontend side, which is exactly what we want.

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
   If it needs an optional column role, declare `requires=("type",)` /
   `("pathways",)` / `("edge_type",)` (roles, not column names —
   `validate_registry` rejects anything else) so it is skipped (not failed)
   on inputs that can't fill the role.
4. Never hardcode column names, layer names, layer counts, or edge-type
   vocabularies: read node attrs `layer` / `type` / `pathways` and edge attr
   `edge_type`, use `c.layer_order(G)` for the layer list, `Counter` over
   `edge_type` for the type vocabulary, `c.node_key` for ordering, and
   `c.columns(G).<role>` only to name the input column in log text. Handle
   an empty `type` / `edge_type` (`""`) gracefully unless you declared the
   role in `requires`.
5. Smoke-test in isolation first: `python3 run_layouts.py -i IN -o
   SCRATCH_OUT --only <base_name> --no-progress` (see "Testing" below for
   the full checklist).
6. Update `README.md`'s method table (keep it alphabetically sorted by
   `base_name`) and the file list at the bottom.

## Testing / verification checklist

Run after adding or changing any layout:

```bash
IN=/path/to/input_dir; OUT=/tmp/layout_smoke
python3 run_layouts.py -i "$IN" --dry-run                        # graph loads, registry validates
python3 run_layouts.py -i "$IN" -o "$OUT" --only <base_name> --no-progress   # isolated smoke test
```

Then verify the write, e.g.:

```python
import math
import common as c
header_in, rows_in = c.read_nodes_tsv(f"{IN}/nodes.tsv")
header, rows = c.read_nodes_tsv(f"{OUT}/nodes.tsv")
assert [r["id"] for r in rows] == [r["id"] for r in rows_in]   # rows unchanged, same order
assert header[:len(header_in)] == header_in                     # input columns untouched, in place
assert len(header) == len(set(header))                          # no duplicate columns
bases = c.get_existing_layout_bases(header)
assert "<base_name>" in bases                                   # frontend would detect it
bad = [(r["id"], col) for r in rows for col in header
       if col.startswith(("x_", "y_", "z_")) and not math.isfinite(float(r[col]))]
assert not bad                                                  # no NaN/Inf
assert c.detect_line_terminator(f"{OUT}/nodes.tsv") == c.detect_line_terminator(f"{IN}/nodes.tsv")
```

Also run the same `--only` smoke test once with a dataset that has
**different column names, string ids and no optional columns** (a tiny
synthetic one is fine) using the `--*-column` flags — that is what catches a
stray `row["id"]`, `int(n)` or `"edge_type_c"` literal.

Also worth a spot-check for genuinely new signals: confirm the column
isn't degenerate (e.g. `domaintouch_1`'s z has at most as many distinct
values as there are layers, by design — check the actual `nunique` matches
what the metric *should* produce, not just "some values").

The input folder must be byte-identical after a run (`git status` / `cmp`
against a copy) — nothing in this directory may write there.

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

- **Never feed a `G.subgraph(...)` view to a layout algorithm; use
  `c.ordered_subgraph(G, nodes)`.** A networkx subgraph view iterates in
  `set` order whenever the kept set is smaller than half the graph
  (`nx.filters.show_nodes` stores a set), and `str` hashes are randomised
  per process — so `nx.forceatlas2_layout(view, seed=42)` hands out its
  seeded random initial positions to different nodes on every run.
  `fa2sep_1`, `community_1`, `community_shellz_1` and `n2vlayered_1` were
  silently non-reproducible for exactly this reason until every
  algorithm-feeding subgraph went through `ordered_subgraph` (confirmed by
  running the old code twice and diffing). `ordered_subgraph` also copies
  each node's *neighbour* order from the parent graph instead of going
  through `add_edges_from` / `Graph.copy()` — those reorder neighbours, and
  the random-walk layouts (node2vec family, `metapath_1`) pick the next step
  by position in `G.neighbors()`, so a plain copy changes their output even
  for the full graph. Views are still fine for `sorted_components` /
  `connected_components`, whose results are sorted anyway. The tell-tale:
  a seeded layout whose coordinates differ between two otherwise identical
  runs.

- **Never feed a raw spectral embedding to ForceAtlas2 as a warm start.**
  Structurally equivalent nodes (same neighbour set) get numerically
  identical spectral coordinates (min pairwise distance ~1e-17 observed on
  the current data), and networkx's FA2 repulsion is 1/d², so the layout
  explodes to ~1e8 units and prints `RuntimeWarning: invalid value
  encountered in divide` from `networkx/drawing/layout.py`. Pass any seed
  through `c.prepare_seed_positions()` (rescale + deterministic jitter)
  first — that is what `fa2spectral_1` does. A tell-tale sign in the
  verification spot-check is a span many orders of magnitude larger than
  the unseeded `fa2_1`.

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
