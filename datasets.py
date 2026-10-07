"""Datasets offered in the app's dropdown.

Each dataset is data/<file>.h5ad (or data/<file>.partKofN.h5ad pieces, joined by cells on load) with:
  X     log-normalized expression (cells x genes, sparse)
  obs   one categorical column per entry in "groupings" (category order = display order)
  obsm  2-D embeddings named in "embeddings"
To add a dataset: write such a file (see prepare_*.py) and add an entry below.
"""

DATASETS = {
    "stromal": {
        "file": "stroma",
        "title": "Mouse Bone Marrow Stroma in Homeostasis",
        "groupings": ["Cluster", "Subcluster"],
        "grouping_help": "Subclusters split clusters 1, 7, 8 and 12 into finer populations.",
        "embeddings": {"UMAP": "X_umap", "tSNE (authors)": "X_tsne"},
        "default_genes": ["Lepr", "Cxcl12", "Kitl", "Vcam1"],
        "default_group_a": {"Cluster": "1", "Subcluster": "1_0"},
        "units": "TP4K",
        "about": """
**Data:** Broad Single Cell Portal study SCP361, *Mouse Bone Marrow Stroma in Homeostasis* (Baryawno et al.).
Expression values are the authors' log-normalized TP4K matrix; cluster and subcluster labels and the
tSNE are the authors'. The UMAP was computed for this app (2,000 highly variable genes, 40 PCs, 15 neighbors).
""",
    },
    "dolgalev": {
        "file": "dolgalev",
        "title": "Bone Marrow Niche Atlas (Dolgalev) — all cells",
        "groupings": ["Harmonized label", "Leiden cluster", "Source dataset"],
        "grouping_help": "Harmonized label: the atlas's cell-type annotation. Leiden cluster: our clustering "
                         "(resolution 0.3). Source dataset: the study each cell came from.",
        "embeddings": {"UMAP": "X_umap", "Atlas embedding": "X_atlas"},
        "default_genes": ["Cdh5", "Lepr", "Cxcl12", "Kitl"],
        "default_group_a": {"Harmonized label": "MSPC-Adipo", "Leiden cluster": "0", "Source dataset": "Baryawno"},
        "units": "CP10k",
        "about": """
**Data:** the Dolgalev bone marrow niche atlas, which harmonizes non-hematopoietic bone marrow cells from
the Baryawno, Tikhonova and Baccin studies (32,743 cells). Harmonized labels and the "Atlas embedding" come
from the atlas metadata. Expression is the atlas's log-normalized matrix (about log(1 + counts per 10k)).
Leiden clusters (resolution 0.3) and the UMAP come from the HSCnicheRNAseq pipeline
(2,000 highly variable genes, 50 PCs, 15 neighbors).

The Baryawno cells here are the same cells as the **stromal** dataset, but normalized differently, so
expression values are not directly comparable between the two.
""",
    },
    "dolgalev endothelial": {
        "file": "dolgalev_endothelial",
        "title": "Bone Marrow Niche Atlas (Dolgalev) — endothelial cells",
        "groupings": ["EC subtype", "Leiden cluster", "Harmonized label", "Source dataset"],
        "grouping_help": "EC subtype: arterial / venous / sinusoidal, assigned per endothelial Leiden cluster "
                         "from marker genes. Leiden cluster: re-clustering of endothelial cells only.",
        "embeddings": {"UMAP": "X_umap", "Atlas embedding": "X_atlas"},
        "default_genes": ["Kitl", "Gja5", "Aplnr", "Stab2"],
        "default_group_a": {"EC subtype": "Arterial", "Leiden cluster": "4", "Harmonized label": "EC-Arteriolar"},
        "units": "CP10k",
        "about": """
**Data:** endothelial cells from the Dolgalev atlas (10,530 cells), selected as in `endothelial_notebook.ipynb`:
atlas Leiden clusters 2, 10 and 12, excluding cells the atlas labels "C" or "Schwann-cells".

**Re-analysis:** genes detected in ≥3 endothelial cells were re-scaled, then PCA (50 PCs), a 15-neighbor graph,
Leiden clustering (resolution 0.5) and UMAP were recomputed on these cells only. Expression values are the
atlas's log-normalized values, unchanged.

**EC subtype:** each cell gets a score for three marker sets (scanpy `score_genes`):
arterial *Gja5, Bmx, Hey1, Hey2, Efnb2, Dll4*; venous *Nr2f2, Ephb4, Aplnr, Vwf, Nt5e, Flrt2*;
sinusoidal *Stab2, Stab1, Mrc1, Fcgr2b, Lyve1, Cd36*. Each Leiden cluster takes the subtype whose average
score is highest relative to other clusters (z ≥ 0.5); clusters with no clear winner are labeled **Other**.
"Other" includes a Ly6a⁺/Cd34⁺ capillary cluster with mixed arterial and sinusoidal features, plus small
proliferating, lymphatic-like, stromal-contaminated and low-quality clusters. Use the Leiden cluster grouping
to look at them separately.
""",
    },
}
