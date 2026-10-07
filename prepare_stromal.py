"""Convert the SCP361 download into a compact sparse .h5ad (with UMAP) for the Streamlit app.

Run once from the SCP361 folder:  python streamlit_app/prepare_stromal.py
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data" / "stroma.h5ad"
MIN_CELLS = 3  # drop genes detected in fewer cells; they can't be tested meaningfully

# Expression file is genes x cells, already log-normalized; stream it to avoid a dense 4+ GB frame.
blocks, genes = [], []
reader = pd.read_csv(ROOT / "expression/stroma.TP4K.txt", sep="\t", index_col=0,
                     chunksize=2000, dtype={0: str}, engine="c")
for chunk in reader:
    vals = np.round(chunk.to_numpy(dtype=np.float32), 3)  # 3 decimals: no visible loss, much better compression
    blocks.append(sp.csr_matrix(vals))
    genes.extend(chunk.index)
    cells = chunk.columns
    print(f"read {len(genes)} genes", flush=True)

X = sp.vstack(blocks).T.tocsr().astype(np.float32)

meta = pd.read_csv(ROOT / "metadata/stroma.tsne.meta.txt", sep="\t", index_col=0, skiprows=[1])
tsne = pd.read_csv(ROOT / "cluster/stroma.tsne.txt", sep="\t", index_col=0, skiprows=[1])
meta = meta.loc[cells].astype(str)

adata = ad.AnnData(X=X, obs=meta, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
adata.var_names_make_unique()
adata.obsm["X_tsne"] = tsne.loc[cells, ["X", "Y"]].to_numpy(dtype=np.float32)

sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

order = sorted(adata.obs["Cluster"].unique(), key=int)
adata.obs["Cluster"] = pd.Categorical(adata.obs["Cluster"], categories=order)
sub_order = sorted(adata.obs["Subcluster"].unique(),
                   key=lambda s: tuple(int(p) for p in s.split("_")))
adata.obs["Subcluster"] = pd.Categorical(adata.obs["Subcluster"], categories=sub_order)

# UMAP is not provided by the authors; compute it from the same log-normalized data.
tmp = adata.copy()
sc.pp.highly_variable_genes(tmp, n_top_genes=2000)
tmp = tmp[:, tmp.var["highly_variable"]].copy()
sc.pp.scale(tmp, max_value=10)
sc.tl.pca(tmp, n_comps=40, random_state=0)
sc.pp.neighbors(tmp, n_neighbors=15, random_state=0)
sc.tl.umap(tmp, random_state=0)
adata.obsm["X_umap"] = tmp.obsm["X_umap"].astype(np.float32)

OUT.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(OUT, compression="gzip", compression_opts=9)
print(adata)
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
