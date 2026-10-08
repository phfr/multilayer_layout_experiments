# multilayer layout generator

Computes a batch of 3D graph layouts for a multilayer network given as an
input folder with `nodes.tsv` + `edges.tsv`, and bakes each one into the
**output folder's** `nodes.tsv` as an `x_<name>`, `y_<name>`, `z_<name>`
column triplet — the naming convention the viewer app auto-detects as a
selectable layout (see `src/data/LayoutDetection.ts`). The input folder is
never modified. Which columns hold the node id, the layer, the optional
per-node type / pathway annotation, the edge endpoints and the optional
edge type is configurable on the command line, so any tab-separated
node/edge export works — the defaults match the wwiznet transcript /
protein / metabolite network.

## Dependencies

Runs against the machine's global Python interpreter (no project-local venv
in this repo, matching the existing `utils/*.py` convention) with
`networkx`, `scikit-learn`, `scipy`, `numpy`, `umap-learn`, `node2vec`,
`gensim`, `python-louvain`, `tqdm`, `phate` (for `n2vbal_phate_1`),
`pacmap` (for the `n2v*_pacmap_1` layouts), and `openTSNE` (for
`n2vbal_tsne_1`) — all already installed. If starting fresh: `pip install
networkx scikit-learn umap-learn node2vec python-louvain tqdm phate pacmap
openTSNE`.

Known issue: running PaCMAP (numba-JIT-compiled) concurrently with another
heavy node2vec/UMAP process can starve it enough to kill it silently (no
error, no "done" line, just a truncated log) — this was resource contention,
not a real bug; run `--only` subsets one at a time if you see this.

## Run it

```bash
python3 run_layouts.py -i /path/to/input_dir -o /path/to/output_dir
```

- `-i / --input-dir` — folder containing `nodes.tsv` and `edges.tsv`. Read
  only.
- `-o / --output-dir` — created if missing. Receives `nodes.tsv` (input
  columns + layout columns), a verbatim copy of `edges.tsv` (so the folder
  is a complete dataset you can point the viewer at), and
  `layout_run_log.txt`.
- `--layer-order transcript,protein,metabolite` — the values of the layer
  column, bottom/innermost first; must list exactly the values present in
  the data. Without it the layers are stacked by **descending node count**
  (ties by first appearance), which on the wwiznet data gives exactly
  transcript / protein / metabolite. The effective order and whether it
  was inferred are printed and logged.

### Telling it which column is which

Every layout reads the graph through column *roles*, never through literal
column names, and each role is mapped to an input column with a flag
(defaults in parentheses):

| Flag | Role | File | Default | Used by |
|---|---|---|---|---|
| `--id-column` | node id | nodes.tsv | `id` | everything |
| `--layer-column` | layer (categorical) | nodes.tsv | `layer_c` | every layer-aware layout |
| `--type-column` | finer per-node type, optional | nodes.tsv | `type_a` | `hive5_1`, `landscape_1` |
| `--pathway-column` | comma-separated pathway annotation, optional | nodes.tsv | `pathways_a` | `fa2pathway_1`, `pathway_landscape_1` |
| `--source-column` / `--target-column` | edge endpoints | edges.tsv | `source` / `target` | everything |
| `--edge-type-column` | categorical edge type, optional | edges.tsv | `edge_type_c` | `fa2rarity_1`, `raritytouch_1`, `shell_rarity_1`, `landscape_1` |

Pass `none` to an optional flag to say the dataset has no such column. A
layout that needs an optional role the input cannot fill (no column
configured, column absent, or empty in every row) is **skipped** with a
`SKIPPED` log entry, never failed; `--list` shows which layouts require
which role and `--dry-run` shows what would be skipped for a given input.
Node ids may be any string (numeric-looking ids still sort numerically).

```bash
# a dataset exported with other column names and no edge types
python3 run_layouts.py -i IN -o OUT --id-column node_id --layer-column tissue_c \
    --source-column from --target-column to --type-column none --edge-type-column none \
    --layer-order gene,mirna,lipid,drug
```

Useful flags:

```bash
python3 run_layouts.py --list                                  # print the registry, no reads/writes
python3 run_layouts.py -i IN --dry-run                         # build the graph, validate, show what would run/skip
python3 run_layouts.py -i IN -o OUT --only fa2_1,mds_1         # run just a subset (comma-separated base names)
python3 run_layouts.py -i IN -o OUT --no-progress              # plain log output, no progress bars
```

Running again into the **same** output folder keeps the layouts already
there and adds/overwrites only the ones you run — the output `nodes.tsv` is
rebuilt from the input's columns plus the existing layout columns (matched
by node id), so `--only` runs accumulate. Pointing `-o` at a folder that
holds a *different* dataset's `nodes.tsv` is refused.

## Tuning UMAP parameters

`evaluate_umap_params.py` grid-searches `n_neighbors`/`min_dist`/`spread`
for the node2vec embeddings and scores each combo on two axes: **trustworthiness**
(does the 3D embedding stay faithful to the original 64-dim node2vec
neighborhood structure, via `sklearn.manifold.trustworthiness`) and a
**clumpiness metric** (coefficient of variation of nearest-neighbor
distances in the 3D output — directly measures "tight groups with big gaps
between them"). Run it before hand-tuning UMAP params by eye:

```bash
python3 evaluate_umap_params.py -i IN                  # evaluates for n2vbal (p=1, q=1)
python3 evaluate_umap_params.py -i IN --p 0.25 --q 4   # evaluate for a different node2vec bias
```

Current defaults (`N2V_UMAP_NEIGHBORS=15, N2V_UMAP_MIN_DIST=0.5,
N2V_UMAP_SPREAD=2.0` in `common.py`) come from this evaluation on the
previous (900-node) dataset: trustworthiness stayed ~flat (0.96-0.99) across
the whole grid, while raising `min_dist` from the old default of 0.15 to
0.5 (and adding `spread=2.0`) roughly halved the clumpiness metric for
negligible fidelity loss. Worth re-running on a new dataset.

## Output and log

Re-running a layout **overwrites its 3 columns in place** in the output
`nodes.tsv` (same column position, no duplicates) — this is intentional and
safe; the log records it as `(recomputed)`. The input columns are never
touched or reordered; new layout columns are always appended at the end.
Line endings (LF or CRLF) are preserved from the input file.

`layout_run_log.txt` in the output folder is append-only. Every run opens
with a banner whose first line is the exact command that was run (so a
result can always be traced back to its invocation), followed by the
interpreter, working directory, input/output paths, graph statistics, layer
order and the list of selected and skipped layouts; then one timestamped
entry per layout with its exact parameters, seed, weighting scheme, and any
disconnected-graph fallback notes. Layouts whose required optional column
role is missing from the input (on the wwiznet data: the two pathway-driven
ones) are recorded as `SKIPPED`, not failed.

Progress: a global bar tracks overall batch progress; each individual layout
shows either its own native progress (node2vec's walk-generation bars, UMAP,
MDS's per-iteration stress) or a small elapsed-time spinner where the
underlying algorithm has no progress hooks (FA2, spring, Kamada-Kawai,
spectral, shell, centrality).

## The graph

Current data (`wwiznet_ppimetabolite/prepare/raw_input`): 263 nodes, 317
edges, a single connected component, no isolated nodes. Each node's layer
comes from the `layer_c` column (the default `--layer-column`) —
**transcript** (141), **protein** (90), **metabolite** (32) — stacked in
that order (transcript bottom/innermost, metabolite top/outermost) by every
layer-aware layout; `--layer-order` changes it. `type_a` (the default
`--type-column`) gives a finer split within a layer (e.g. `protein_bridge`
vs `protein_differentially_expressed`) that `hive5_1` and `landscape_1`
use. Edge types (`edge_type_c`, the default `--edge-type-column`):
protein-protein, transcript-transcript, metabolite-metabolite,
protein-metabolite, transcript-metabolite — there are no direct
transcript-protein edges, so the two outer layers only meet through
metabolites. Nothing in the code depends on these particular values: layer
names, type values and edge-type vocabularies are all read from the data.

Every method handles the disconnected structure explicitly — either by
excluding isolated nodes and placing them on a deterministic ring/sphere
afterward (force-directed methods), or by running only on the giant
component and placing everything else (isolated + small components) on a
deterministic Fibonacci-sphere shell (methods that depend on global
graph-distance or embedding structure). See `common.py` for the placement
helpers.

## Methods

| base name(s) | family | what it does |
|---|---|---|
| `community_1` | Louvain community | Detects communities via `python-louvain` (emergent, not the a priori layer label — a community can span multiple layers), lays out the community meta-graph with ForceAtlas2, then locally arranges each community's members around its slot. |
| `community_shellz_1` | Community + shell blend | `community_1`'s real force-computed xy (unchanged) + z from `shell_1`'s radius-by-layer formula (signed by the layer's position in `--layer-order`), applied per **community's predominant layer** rather than per node — "communities sorted by predominant measurement layer". No Procrustes alignment needed since it reuses `shell_1`'s radius *formula*, not its coordinates. Logs how many communities are single-layer-pure vs. genuinely mixed (mixed ones get a lossy majority-vote z). |
| `domaintouch_1` | Centrality elevation | Reuses `fa2_1`'s xy, z = number of distinct layers a node's own layer **plus its 1-hop neighborhood** collectively touch (1 to the number of layers) — catches low-degree nodes that quietly bridge multiple layers, a signal neither the layer label nor centrality alone expresses. |
| `ensemble_1` | Procrustes blend | Generalized Procrustes alignment (Gower 1975) of `fa23d_1` + `mds_1` + `spectral_1` — three deliberately different objectives (local force clustering, global geodesic fidelity, algebraic connectivity) aligned via `scipy.linalg.orthogonal_procrustes` and weighted-averaged into one consensus layout. |
| `fa23d_1` | ForceAtlas2 | Native 3D ForceAtlas2 (no manual z-stack — all 3 axes come from the force simulation). |
| `fa23d_2` | ForceAtlas2 | Same as `fa23d_1` but cross-layer edges down-weighted, so layers separate organically in 3D. |
| `fa2_1` | ForceAtlas2 | Shared 2D force layout (xy) for the whole graph, with a fixed z plane per layer, stacked in `--layer-order` (transcript / protein / metabolite bottom to top). |
| `fa2_2` | ForceAtlas2 | Same as `fa2_1` but with `linlog=True, dissuade_hubs=True` — compresses hub-dominated clusters differently. |
| `fa2_3` | ForceAtlas2 | Same as `fa2_1` but cross-layer edges are down-weighted before layout, so layers spread apart in xy *before* the z-stack is applied. |
| `fa2pathway_1` | ForceAtlas2 | Real interaction edges get their attraction boosted by the Jaccard similarity of their endpoints' pathway annotation (`--pathway-column`) — real topology stays primary, pathway co-membership only reinforces it. Unannotated nodes/edges behave exactly like `fa2_1`. **Skipped** when the input has no pathway column (the current data). |
| `fa2rarity_1` | ForceAtlas2 | Per-edge-type (`--edge-type-column`) rarity weighting (-ln(count/total) per exact type, one bucket per distinct type found in the data, not the binary within/cross-layer split `fa2_3` uses) — edges of a rare type pull tighter than edges of a common one. |
| `fa2sep_1` | ForceAtlas2 | Each layer's induced subgraph (same-layer edges only) is laid out *independently*, then z-stacked in `--layer-order`. xy is left overlapping across layers (no artificial offset). |
| `fa2spectral_1` | ForceAtlas2 | Seeded with a spectral embedding instead of random init (the fCoSE trick from Cytoscape's Compound Spring Embedder) — usually better global structure / faster convergence than `fa2_1`. |
| `fr3d_1` | Fruchterman-Reingold | Native 3D spring layout (`nx.spring_layout`, unweighted). |
| `fr3d_2` | Fruchterman-Reingold | Same, with cross-layer edges down-weighted. |
| `hive3_1` | Hive plot | One radial spoke per layer (Krzywinski-style hive plot, adapted to 3D via `fibonacci_sphere(n_layers)`), nodes ordered by degree along their axis (hubs near the shared origin). Pure deterministic geometry, no simulation. |
| `hive5_1` | Hive plot | Finer variant of `hive3_1` — one spoke per raw value of the type column (`--type-column`; currently 5: `protein_bridge`, `protein_differentially_expressed`, `transcript_bridge`, `transcript_differentially_expressed`, `metabolite_differentially_expressed`), spokes ordered by layer then name so a layer's sub-types sit on adjacent axes. Axis count is data-driven; the name keeps its historical `5`. |
| `hyp_1` | Hyperbolic radius | `r = 1 - tanh(degree/beta)` (inspired by "Hyperbolic Embedding of Multilayer Networks", arXiv:2505.20378) — degree-0 nodes land exactly on the boundary automatically, no fallback shell needed. Azimuth from `fa2_1`'s xy, polar angle banded by layer (one latitude band per layer, evenly spread between 30 and 150 degrees in `--layer-order`). |
| `isomap_1` | Isomap | Kernel PCA on a k-NN-restricted geodesic distance matrix (giant component, `sklearn.manifold.Isomap`). A distinct embedding objective from both MDS (stress majorization on the full distance matrix) and spectral (Laplacian eigenmaps). |
| `kk3d_1` | Kamada-Kawai | Path-length-based 3D layout, computed on the giant component only; every other node placed via the deterministic fallback shell. Unweighted hop distance. |
| `kk3d_2` | Kamada-Kawai | Same, but cross-layer edges get 3x the hop-distance weight, explicitly pushing layers apart. |
| `landscape_1` | Feature-space embedding | VRNetzer-style "functional landscape" (Pfeil et al., *Nat. Commun.* 2021): per-node feature vector (edge-type composition fractions + layer one-hot + node-type one-hot + Louvain community one-hot + log-degree; the edge-type and type blocks are simply absent when the input has no such column) embedded via UMAP. Similarity is in **annotation space**, not graph-distance space — two nodes with similar profiles land close together even across different connected components. |
| `mds_1` | Classical MDS | Metric MDS on the giant component's shortest-path distance matrix. Unweighted hop count. |
| `mds_2` | Classical MDS | Same, with cross-layer hop length inflated (3x) before the distance matrix is built. |
| `metapath_1` | node2vec-style, typed | Custom random-walk generator biased 4x toward staying within the same layer (both endpoints in the same layer: protein-protein, metabolite-metabolite, etc.) before crossing into another layer — node2vec's own p/q never look at node type at all. Needs no edge-type column (same-domain is defined from the endpoints' layers, which coincides exactly with the `edge_type_c` convention on the wwiznet data). Feeds directly into gensim's Word2Vec + UMAP. |
| `n2vbal_1` | node2vec + UMAP | node2vec random-walk embedding (p=1, q=1: balanced walks) on the giant component, reduced to 3D via UMAP (cosine metric). |
| `n2vbal_densmap_1` | node2vec + DensMAP | Same node2vec embedding as `n2vbal_1`, reduced via DensMAP (Narayan, Berger & Cho, *Nat. Biotechnol.* 2021) — already free in the installed `umap-learn` via `densmap=True`; preserves true local density instead of UMAP's default behavior of visually equalizing it everywhere. |
| `n2vbal_pacmap_1` | node2vec + PaCMAP | Same node2vec embedding as `n2vbal_1`, reduced via PaCMAP (Wang et al., JMLR 2021) instead of UMAP — the same technique the frontend already uses client-side for its own in-browser layout calc. Often more evenly spread than UMAP by design (balances near/mid-near/further point-pair sampling). Requires the `pacmap` package. |
| `n2vbal_phate_1` | node2vec + PHATE | Same node2vec embedding as `n2vbal_1`, reduced via PHATE (Moon et al., *Nat. Biotechnol.* 2019) instead of UMAP — PHATE is purpose-built for biological data, preserving both local *and* global/trajectory structure. Requires the `phate` package. |
| `n2vbal_tsne_1` | node2vec + openTSNE | Same node2vec embedding as `n2vbal_1`, reduced via classic t-SNE (van der Maaten & Hinton 2008) through `openTSNE` (actively maintained, much faster than sklearn's) instead of UMAP — tends to produce tighter, more visually separated clusters than UMAP/PaCMAP give on this data, at some cost to inter-cluster distance fidelity. Uses `negative_gradient_method='bh'` (Barnes-Hut) since openTSNE's FFT acceleration is 2D-only. Requires the `openTSNE` package. |
| `n2vbfs_1` | node2vec + UMAP | Same, p=0.25, q=4 — more BFS-like/homophily-biased walks (emphasizes local neighborhood structure). |
| `n2vbfs_pacmap_1` | node2vec + PaCMAP | Same as `n2vbal_pacmap_1` but p=0.25, q=4 (BFS-like walks). |
| `n2vdfs_1` | node2vec + UMAP | Same, p=4, q=0.25 — more DFS-like/structural-equivalence-biased walks (emphasizes role/bridging structure). |
| `n2vdfs_pacmap_1` | node2vec + PaCMAP | Same as `n2vbal_pacmap_1` but p=4, q=0.25 (DFS-like walks). |
| `n2vlayered_1` | node2vec + UMAP, layered | node2vec+UMAP run **separately** on the first and last layer's same-layer subnetworks (transcript and metabolite with the default order), each z-squeezed into a thin slab and offset apart (first layer negative z, last positive z). Nodes of the middle layer(s) (protein) are placed via Tutte/harmonic-style iterative relaxation — each one's position converges to the average position of its graph neighbors, propagated from the two fixed slabs. Note that with the current data proteins have no transcript neighbors, so they relax toward the metabolite slab and their own protein-protein neighbors; protein components with no metabolite neighbor at all fall back to a Fibonacci-sphere shell around the combined centroid. A slab layer whose giant component is under 10 nodes gets a 3D spring layout instead of node2vec+UMAP. |
| `paga_1` | Two-level coarse+fine | PAGA-style layout (Wolf et al., *Genome Biology* 2019): Louvain communities abstracted into a meta-graph weighted by observed-vs-expected cross-community connectivity (configuration-model null), laid out with ForceAtlas2, then every node initialized near its community's coarse position and refined with a full spring-layout pass over the real graph. Community structure drives *initialization* for real physics, unlike `community_1`'s fixed placement. |
| `pathway_landscape_1` | Feature-space embedding | The real-biology counterpart to `landscape_1`: TF-IDF-weighted pathway-annotation multi-hot (`--pathway-column`) → `TruncatedSVD` → UMAP(3D). Unannotated nodes get an inferred profile via 1-2 hop neighbor mean-pooling (a metabolite infers a profile from the proteins it interacts with); nodes with no annotated node within 2 hops fall back to the usual Fibonacci-sphere shell. **Skipped** when the input has no pathway column (the current data). |
| `raritytouch_1` | Centrality elevation | Reuses `fa2_1`'s xy, z = -ln(rarest edge-type count a node touches / total edges) — the opposite signal from `topodeg_1`/`topobet_1`: a low-degree node with one rare cross-domain edge outranks a hub whose edges are all common types. |
| `rwr_1` | Feature-space embedding | Same paradigm as `landscape_1` but the feature vector is a personalized-PageRank (random-walk-with-restart, r=0.9) visitation distribution per node — a diffusion-based feature space instead of a fixed annotation vector. |
| `shell_1` | Concentric shells | Pure deterministic geometry (no simulation): one spherical shell per layer in `--layer-order` (transcript innermost, then protein, metabolite outermost), radii scaled by node count. Node order within each shell groups same-component nodes together. |
| `shell_2` | Concentric shells | Same, with the radius order reversed (metabolite inner, transcript outer). |
| `shell_rarity_1` | Concentric shells | Self-contained shell family (like `shell_1`/`shell_2`) but banded by the rarest edge type (`--edge-type-column`) a node touches, not its layer — one band per distinct edge-type frequency found in the data, rarest innermost, so rare cross-domain connectors form the innermost shell regardless of which layer they belong to; isolated nodes get their own outermost band. |
| `spectral_1` | Spectral embedding | Laplacian eigenmaps on the giant component's adjacency matrix (unweighted), with a small epsilon background added for numerical stability. |
| `spectral_2` | Spectral embedding | Same, with cross-layer adjacency entries down-weighted. |
| `topobet_1` | Centrality elevation | Reuses `fa2_1`'s xy, sets z from betweenness centrality. |
| `topodeg_1` | Centrality elevation | Reuses `fa2_1`'s xy, sets z from degree centrality ("importance elevation"). |
## Files

- `common.py` — shared TSV IO (line-terminator-preserving), output-folder
  seeding, the `ColumnSpec` column-role model and its CLI flags, graph
  building from the layer column + `--layer-order`, edge weighting
  (two conventions: attraction-style for FA2/spring, distance-style for
  Kamada-Kawai/MDS), disconnected-component analysis, deterministic
  ring/shell placement, log writers (run banner + per-layout entry),
  progress spinner.
- `method_forceatlas.py`, `method_classic_force.py`,
  `method_distance_embedding.py`, `method_node2vec.py`, `method_shell.py`,
  `method_centrality.py`, `method_hyperbolic.py`, `method_hive.py`,
  `method_community.py`, `method_layered_n2v.py`, `method_landscape.py`,
  `method_paga.py`, `method_metapath.py`, `method_ensemble.py`,
  `method_pathway.py` — one module per method family.
- `run_layouts.py` — CLI entry point (`-i`/`-o`/`--layer-order`/`--*-column`)
  and the run registry (add a new `RunSpec` here to register another layout;
  declare `requires=("type",)` / `("pathways",)` / `("edge_type",)` for
  layouts that need an optional column role).
- `evaluate_umap_params.py` — read-only UMAP parameter grid search (`-i`,
  same column flags as `run_layouts.py`).
