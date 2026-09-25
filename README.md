# ppicml layout generator

Computes a batch of 3D graph layouts for the ppicml protein/metabolite
multilayer network (`public/data/ppicml/nodes.tsv` + `edges.tsv`) and bakes
each one into `nodes.tsv` as an `x_<name>`, `y_<name>`, `z_<name>` column
triplet — the naming convention the viewer app auto-detects as a selectable
layout (see `src/data/LayoutDetection.ts`).

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

From anywhere in the repo (paths resolve relative to this script, not your
cwd):

```bash
python3 utils/ppicml_layouts/run_layouts.py
```

or equivalently, from this directory:

```bash
cd utils/ppicml_layouts
python3 run_layouts.py
```

Useful flags:

```bash
python3 run_layouts.py --list                 # print the registry, no writes
python3 run_layouts.py --dry-run               # build the graph, validate names, no writes
python3 run_layouts.py --only fa2_1,mds_1       # run just a subset (comma-separated base names)
python3 run_layouts.py --no-progress            # plain log output, no progress bars
```

## Tuning UMAP parameters

`evaluate_umap_params.py` grid-searches `n_neighbors`/`min_dist`/`spread`
for the node2vec embeddings and scores each combo on two axes: **trustworthiness**
(does the 3D embedding stay faithful to the original 64-dim node2vec
neighborhood structure, via `sklearn.manifold.trustworthiness`) and a
**clumpiness metric** (coefficient of variation of nearest-neighbor
distances in the 3D output — directly measures "tight groups with big gaps
between them"). Run it before hand-tuning UMAP params by eye:

```bash
python3 evaluate_umap_params.py                      # evaluates for n2vbal (p=1, q=1)
python3 evaluate_umap_params.py --p 0.25 --q 4         # evaluate for a different node2vec bias
```

Current defaults (`N2V_UMAP_NEIGHBORS=15, N2V_UMAP_MIN_DIST=0.5,
N2V_UMAP_SPREAD=2.0` in `common.py`) come from this evaluation: trustworthiness
stays ~flat (0.96-0.99) across the whole grid, while raising `min_dist` from
the old default of 0.15 to 0.5 (and adding `spread=2.0`) roughly halves the
clumpiness metric for negligible fidelity loss.

Re-running a layout **overwrites its 3 columns in place** (same column
position, no duplicates) — this is intentional and safe; the log records it
as `(recomputed)`. The original 8 nodes.tsv columns are never touched or
reordered; new layout columns are always appended at the end.

Every run appends a timestamped entry to `layout_run_log.txt` (in
`public/data/ppicml/`) with its exact parameters, seed, weighting scheme, and
any disconnected-graph fallback notes.

Progress: a global bar tracks overall batch progress; each individual layout
shows either its own native progress (node2vec's walk-generation bars, UMAP,
MDS's per-iteration stress) or a small elapsed-time spinner where the
underlying algorithm has no progress hooks (FA2, spring, Kamada-Kawai,
spectral, shell, centrality).

## The graph

900 nodes, 1689 edges, 118 connected components (one giant component of 745
nodes, 14 small components of 2-10 nodes, 103 fully isolated/degree-0
nodes). Nodes are grouped into 3 conceptual layers (`layer3`, derived from
`type_a`): **protein** (protein, transcript, protein+transcript), **bridge**
(genes with no protein/transcript/metabolite measurement), **metabolite**.

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
| `community_1` | Louvain community | Detects communities via `python-louvain` (emergent, not the a priori layer3 label — a community can span multiple layers), lays out the community meta-graph with ForceAtlas2, then locally arranges each community's members around its slot. |
| `community_shellz_1` | Community + shell blend | `community_1`'s real force-computed xy (unchanged) + z from `shell_1`'s radius-by-layer3 formula, applied per **community's predominant layer3** rather than per node — "communities sorted by predominant measurement layer". No Procrustes alignment needed since it reuses `shell_1`'s radius *formula*, not its coordinates. Logs how many communities are single-layer-pure vs. genuinely mixed (mixed ones get a lossy majority-vote z). |
| `domaintouch_1` | Centrality elevation | Reuses `fa2_1`'s xy, z = number of distinct domains (protein/transcript/bridge/metabolite) a node's own type_a **plus its 1-hop neighborhood** collectively touch (1-4) — catches low-degree nodes that quietly bridge multiple domains, a signal neither `type_a` nor centrality alone expresses. |
| `ensemble_1` | Procrustes blend | Generalized Procrustes alignment (Gower 1975) of `fa23d_1` + `mds_1` + `spectral_1` — three deliberately different objectives (local force clustering, global geodesic fidelity, algebraic connectivity) aligned via `scipy.linalg.orthogonal_procrustes` and weighted-averaged into one consensus layout. |
| `fa23d_1` | ForceAtlas2 | Native 3D ForceAtlas2 (no manual z-stack — all 3 axes come from the force simulation). |
| `fa23d_2` | ForceAtlas2 | Same as `fa23d_1` but cross-layer edges down-weighted, so layers separate organically in 3D. |
| `fa2_1` | ForceAtlas2 | Shared 2D force layout (xy) for the whole graph, with a fixed z offset per layer3 (protein / bridge / metabolite stacked as 3 planes). |
| `fa2_2` | ForceAtlas2 | Same as `fa2_1` but with `linlog=True, dissuade_hubs=True` — compresses hub-dominated clusters differently. |
| `fa2_3` | ForceAtlas2 | Same as `fa2_1` but cross-layer edges are down-weighted before layout, so layers spread apart in xy *before* the z-stack is applied. |
| `fa2pathway_1` | ForceAtlas2 | Real interaction edges get their attraction boosted by the Jaccard similarity of their endpoints' real `pathways_a` annotation (321/1689 edges currently get a nonzero boost) — real topology stays primary, pathway co-membership only reinforces it. Unannotated nodes/edges behave exactly like `fa2_1`. |
| `fa2rarity_1` | ForceAtlas2 | Full 10-bucket `edge_type_c` rarity weighting (-ln(count/total) per exact type, not the binary within/cross-layer split `fa2_3` uses) — the 28 rare metabolite-protein edges pull ~3.4x tighter than the 500 common metabolite-metabolite ones. |
| `fa2sep_1` | ForceAtlas2 | Each layer's induced subgraph (protein-only, bridge-only, metabolite-only edges) is laid out *independently*, then z-stacked. xy is left overlapping across layers (no artificial offset). |
| `fa2spectral_1` | ForceAtlas2 | Seeded with a spectral embedding instead of random init (the fCoSE trick from Cytoscape's Compound Spring Embedder) — usually better global structure / faster convergence than `fa2_1`. |
| `fr3d_1` | Fruchterman-Reingold | Native 3D spring layout (`nx.spring_layout`, unweighted). |
| `fr3d_2` | Fruchterman-Reingold | Same, with cross-layer edges down-weighted. |
| `hive3_1` | Hive plot | One radial spoke per layer3 (Krzywinski-style hive plot, adapted to 3 axes in 3D via `fibonacci_sphere(3)`), nodes ordered by degree along their axis (hubs near the shared origin). Pure deterministic geometry, no simulation. |
| `hive5_1` | Hive plot | 5-axis generalization of `hive3_1` — one spoke per raw `type_a` value (protein/transcript/protein,transcript/bridge/metabolite) instead of the coarsened 3-way `layer3`, so the 21 dual-typed nodes get their own visible spoke instead of being folded into "protein". |
| `hyp_1` | Hyperbolic radius | `r = 1 - tanh(degree/beta)` (inspired by "Hyperbolic Embedding of Multilayer Networks", arXiv:2505.20378) — degree-0 nodes land exactly on the boundary automatically, no fallback shell needed. Azimuth from `fa2_1`'s xy, polar angle banded by layer3. |
| `isomap_1` | Isomap | Kernel PCA on a k-NN-restricted geodesic distance matrix (giant component, `sklearn.manifold.Isomap`). A distinct embedding objective from both MDS (stress majorization on the full distance matrix) and spectral (Laplacian eigenmaps). |
| `kk3d_1` | Kamada-Kawai | Path-length-based 3D layout, computed on the giant component only (745/900 nodes); every other node placed via the deterministic fallback shell. Unweighted hop distance. |
| `kk3d_2` | Kamada-Kawai | Same, but cross-layer edges get 3x the hop-distance weight, explicitly pushing layers apart. |
| `landscape_1` | Feature-space embedding | VRNetzer-style "functional landscape" (Pfeil et al., *Nat. Commun.* 2021): per-node feature vector (edge_type_c composition fractions + type_a one-hot + Louvain community one-hot + log-degree) embedded via UMAP. Similarity is in **annotation space**, not graph-distance space — two nodes with similar profiles land close together even across different connected components. |
| `mds_1` | Classical MDS | Metric MDS on the giant component's shortest-path distance matrix. Unweighted hop count. |
| `mds_2` | Classical MDS | Same, with cross-layer hop length inflated (3x) before the distance matrix is built. |
| `metapath_1` | node2vec-style, typed | Custom random-walk generator biased 4x toward staying within the same `edge_type_c` domain (protein-protein, metabolite-metabolite, etc.) before crossing — node2vec's own p/q never look at edge type at all. Feeds directly into gensim's Word2Vec + UMAP. |
| `n2vbal_1` | node2vec + UMAP | node2vec random-walk embedding (p=1, q=1: balanced walks) on the giant component, reduced to 3D via UMAP (cosine metric). |
| `n2vbal_densmap_1` | node2vec + DensMAP | Same node2vec embedding as `n2vbal_1`, reduced via DensMAP (Narayan, Berger & Cho, *Nat. Biotechnol.* 2021) — already free in the installed `umap-learn` via `densmap=True`; preserves true local density instead of UMAP's default behavior of visually equalizing it everywhere. |
| `n2vbal_pacmap_1` | node2vec + PaCMAP | Same node2vec embedding as `n2vbal_1`, reduced via PaCMAP (Wang et al., JMLR 2021) instead of UMAP — the same technique the frontend already uses client-side for its own in-browser layout calc. Often more evenly spread than UMAP by design (balances near/mid-near/further point-pair sampling). Requires the `pacmap` package. |
| `n2vbal_phate_1` | node2vec + PHATE | Same node2vec embedding as `n2vbal_1`, reduced via PHATE (Moon et al., *Nat. Biotechnol.* 2019) instead of UMAP — PHATE is purpose-built for biological data, preserving both local *and* global/trajectory structure. Requires the `phate` package. |
| `n2vbal_tsne_1` | node2vec + openTSNE | Same node2vec embedding as `n2vbal_1`, reduced via classic t-SNE (van der Maaten & Hinton 2008) through `openTSNE` (actively maintained, much faster than sklearn's) instead of UMAP — tends to produce tighter, more visually separated clusters than UMAP/PaCMAP give on this data, at some cost to inter-cluster distance fidelity. Uses `negative_gradient_method='bh'` (Barnes-Hut) since openTSNE's FFT acceleration is 2D-only. Requires the `openTSNE` package. |
| `n2vbfs_1` | node2vec + UMAP | Same, p=0.25, q=4 — more BFS-like/homophily-biased walks (emphasizes local neighborhood structure). |
| `n2vbfs_pacmap_1` | node2vec + PaCMAP | Same as `n2vbal_pacmap_1` but p=0.25, q=4 (BFS-like walks). |
| `n2vdfs_1` | node2vec + UMAP | Same, p=4, q=0.25 — more DFS-like/structural-equivalence-biased walks (emphasizes role/bridging structure). |
| `n2vdfs_pacmap_1` | node2vec + PaCMAP | Same as `n2vbal_pacmap_1` but p=4, q=0.25 (DFS-like walks). |
| `n2vlayered_1` | node2vec + UMAP, layered | node2vec+UMAP run **separately** on the protein-only and metabolite-only same-layer subnetworks, each z-squeezed into a thin slab and offset apart (protein negative z, metabolite positive z). Bridge nodes are placed via Tutte/harmonic-style iterative relaxation — each bridge node's position converges to the average position of its graph neighbors, propagated from the two fixed layers, so a bridge node connecting only to protein neighbors lands near the protein slab, one connecting to both lands truly in between. A handful of bridge-only components with no path to either layer (6/152 currently) fall back to a Fibonacci-sphere shell around the combined centroid. |
| `paga_1` | Two-level coarse+fine | PAGA-style layout (Wolf et al., *Genome Biology* 2019): Louvain communities abstracted into a meta-graph weighted by observed-vs-expected cross-community connectivity (configuration-model null), laid out with ForceAtlas2, then every node initialized near its community's coarse position and refined with a full spring-layout pass over the real graph. Community structure drives *initialization* for real physics, unlike `community_1`'s fixed placement. |
| `pathway_landscape_1` | Feature-space embedding | The real-biology counterpart to `landscape_1`: TF-IDF-weighted `pathways_a` multi-hot → `TruncatedSVD` → UMAP(3D). `pathways_a` only directly annotates 428/900 nodes (zero metabolites — it's gene-centric), so unannotated nodes get an inferred profile via 1-2 hop neighbor mean-pooling (a metabolite infers a profile from the proteins it interacts with), lifting real coverage to 838/900 (93%); the remaining 62 with no annotated node within 2 hops fall back to the usual Fibonacci-sphere shell. |
| `raritytouch_1` | Centrality elevation | Reuses `fa2_1`'s xy, z = -ln(rarest `edge_type_c` count a node touches / total edges) — the opposite signal from `topodeg_1`/`topobet_1`: a low-degree node with one rare cross-domain edge outranks a hub whose edges are all common types. |
| `rwr_1` | Feature-space embedding | Same paradigm as `landscape_1` but the feature vector is a personalized-PageRank (random-walk-with-restart, r=0.9) visitation distribution per node — a diffusion-based feature space instead of a fixed annotation vector. |
| `shell_1` | Concentric shells | Pure deterministic geometry (no simulation): protein on an inner sphere, bridge on a middle shell, metabolite on an outer shell, radii scaled by node count. Node order within each shell groups same-component nodes together. |
| `shell_2` | Concentric shells | Same, with the radius order reversed (metabolite inner, protein outer). |
| `shell_rarity_1` | Concentric shells | New self-contained shell family (like `shell_1`/`shell_2`) but banded by the rarest `edge_type_c` a node touches, not `layer3` — rare cross-domain connectors form the innermost shell regardless of which layer they belong to. |
| `spectral_1` | Spectral embedding | Laplacian eigenmaps on the giant component's adjacency matrix (unweighted), with a small epsilon background added for numerical stability. |
| `spectral_2` | Spectral embedding | Same, with cross-layer adjacency entries down-weighted. |
| `topobet_1` | Centrality elevation | Reuses `fa2_1`'s xy, sets z from betweenness centrality. |
| `topodeg_1` | Centrality elevation | Reuses `fa2_1`'s xy, sets z from degree centrality ("importance elevation"). |
## Files

- `common.py` — shared TSV IO (CRLF-preserving), graph building, `layer3`
  classification, edge weighting (two conventions: attraction-style for
  FA2/spring, distance-style for Kamada-Kawai/MDS), disconnected-component
  analysis, deterministic ring/shell placement, log writer, progress spinner.
- `method_forceatlas.py`, `method_classic_force.py`,
  `method_distance_embedding.py`, `method_node2vec.py`, `method_shell.py`,
  `method_centrality.py`, `method_hyperbolic.py`, `method_hive.py`,
  `method_community.py`, `method_layered_n2v.py`, `method_landscape.py`,
  `method_paga.py`, `method_metapath.py`, `method_ensemble.py`,
  `method_pathway.py` — one module per method family.
- `run_layouts.py` — CLI entry point and the run registry (add a new
  `RunSpec` here to register another layout).
