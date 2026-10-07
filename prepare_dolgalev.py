"""Build the "dolgalev" and "dolgalev endothelial" datasets for the Streamlit app.

Inputs come from the HSCnicheRNAseq pipeline (results/processed_data.h5ad, whose .raw holds the
log-normalized atlas matrix) and the atlas metadata (harmonized labels + published embedding).

    python prepare_dolgalev.py --atlas ~/Documents/HSCnicheRNAseq/results/processed_data.h5ad \
                               --metadata ~/Downloads/metadata.csv
"""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

OUT_DIR = Path(__file__).resolve().parent / "data"
MIN_CELLS = 3
SEED = 0

# Endothelial selection, as in endothelial_notebook.ipynb: atlas Leiden clusters 2/10/12 (res 0.3),
# excluding cells the atlas labels "C" or "Schwann-cells".
ENDO_LEIDEN = ["2", "10", "12"]
EXCLUDED_LABELS = ["C", "Schwann-cells"]

EC_MARKERS = {
    "Arterial": ["Gja5", "Bmx", "Hey1", "Hey2", "Efnb2", "Dll4"],
    "Venous": ["Nr2f2", "Ephb4", "Aplnr", "Vwf", "Nt5e", "Flrt2"],
    "Sinusoidal": ["Stab2", "Stab1", "Mrc1", "Fcgr2b", "Lyve1", "Cd36"],
}
MIN_SUBTYPE_Z = 0.5  # cluster's best marker score must stand out this far (z across clusters) to be labeled


def natural_categorical(values):
    values = pd.Series(values, dtype=str)
    key = lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s)
    return pd.Categorical(values, categories=sorted(values.unique(), key=key))


def write(adata, name, max_mb=95, shard_mb=80):
    """Write data/<name>.h5ad, or split it by cells into <name>.partKofN.h5ad files if it would exceed
    GitHub's per-file limit. Categoricals keep their full category list in every part."""
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    adata.X = adata.X.tocsr().astype(np.float32)
    adata.X.data = np.round(adata.X.data, 3)
    adata.var = adata.var[[]]
    for old in OUT_DIR.glob(f"{name}.part*of*.h5ad"):
        old.unlink()
    path = OUT_DIR / f"{name}.h5ad"
    adata.write_h5ad(path, compression="gzip", compression_opts=9)
    size_mb = path.stat().st_size / 1e6
    print(f"wrote {path.name}: {adata.n_obs:,} cells x {adata.n_vars:,} genes, {size_mb:.1f} MB")
    if size_mb <= max_mb:
        return
    path.unlink()
    n = int(np.ceil(size_mb / shard_mb))
    bounds = np.linspace(0, adata.n_obs, n + 1).astype(int)
    for i in range(n):
        rows = slice(bounds[i], bounds[i + 1])
        # Built directly (not via adata[rows]) because AnnData slicing drops unused categories.
        piece = ad.AnnData(X=adata.X[rows], obs=adata.obs.iloc[rows], var=adata.var,
                           obsm={k: v[rows] for k, v in adata.obsm.items()})
        part = OUT_DIR / f"{name}.part{i + 1}of{n}.h5ad"
        piece.write_h5ad(part, compression="gzip", compression_opts=9)
        print(f"  split -> {part.name} ({part.stat().st_size / 1e6:.1f} MB)")


def annotate_subtypes(log_adata, clusters):
    """Label each cluster by whichever EC marker score is highest relative to the other clusters."""
    scores = pd.DataFrame(index=log_adata.obs_names)
    for subtype, genes in EC_MARKERS.items():
        present = [g for g in genes if g in log_adata.var_names]
        sc.tl.score_genes(log_adata, present, score_name="_s", random_state=SEED)
        scores[subtype] = log_adata.obs.pop("_s")
    per_cluster = scores.groupby(clusters.to_numpy(), observed=True).mean()
    z = (per_cluster - per_cluster.mean()) / per_cluster.std()
    best = z.idxmax(axis=1).where(z.max(axis=1) >= MIN_SUBTYPE_Z, "Other")
    print("cluster marker scores (z across clusters):")
    print(z.round(2).assign(label=best, cells=clusters.value_counts()).to_string())
    return clusters.map(best).astype(str)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", type=Path,
                   default=Path("~/Documents/HSCnicheRNAseq/results/processed_data.h5ad").expanduser())
    p.add_argument("--metadata", type=Path, default=Path("~/Downloads/metadata.csv").expanduser())
    p.add_argument("--reference-labels", type=Path,
                   default=Path("~/Documents/HSCnicheRNAseq/endothelial_scRNAseq_for_yotam.h5ad").expanduser(),
                   help="optional earlier endothelial object with obs['endo_group'], used only for a concordance check")
    args = p.parse_args()

    full = ad.read_h5ad(args.atlas)
    meta = pd.read_csv(args.metadata, skiprows=[1], encoding="utf-8-sig").set_index("NAME").loc[full.obs_names]

    obs = pd.DataFrame(index=full.obs_names)
    obs["Harmonized label"] = natural_categorical(meta["Harmonized Label"].to_numpy())
    obs["Leiden cluster"] = natural_categorical(full.obs["leiden"].to_numpy())
    obs["Source dataset"] = natural_categorical(meta["Source Dataset"].to_numpy())
    atlas = ad.AnnData(X=full.raw.X.tocsr(), obs=obs, var=pd.DataFrame(index=full.raw.var_names))
    atlas.obsm["X_umap"] = full.obsm["X_umap"].astype(np.float32)
    atlas.obsm["X_atlas"] = meta[["X", "Y"]].to_numpy(np.float32)

    rowsum = np.asarray(np.expm1(atlas.X[:200].toarray()).sum(1)).ravel()
    print(f"linear-scale library size of first 200 cells: median {np.median(rowsum):,.0f}")

    is_endo = full.obs["leiden"].astype(str).isin(ENDO_LEIDEN) & ~obs["Harmonized label"].isin(EXCLUDED_LABELS)
    endo = atlas[is_endo.to_numpy()].copy()
    write(atlas, "dolgalev")

    # Re-cluster endothelial cells. Expression stays on the atlas's log-normalized scale
    # (the notebook's extra normalize_total on already-log data is intentionally skipped).
    sc.pp.filter_genes(endo, min_cells=MIN_CELLS)
    tmp = endo.copy()
    sc.pp.highly_variable_genes(tmp, flavor="seurat", n_top_genes=tmp.n_vars, subset=True)
    sc.pp.scale(tmp, max_value=10)
    sc.tl.pca(tmp, n_comps=50, svd_solver="arpack", random_state=SEED)
    sc.pp.neighbors(tmp, n_neighbors=15, n_pcs=50, random_state=SEED)
    sc.tl.leiden(tmp, resolution=0.5, key_added="leiden", flavor="igraph", n_iterations=2, random_state=SEED)
    sc.tl.umap(tmp, random_state=SEED)

    clusters = natural_categorical(tmp.obs["leiden"].to_numpy())
    clusters = pd.Series(clusters, index=endo.obs_names)
    subtype = annotate_subtypes(endo.copy(), clusters)
    order = [s for s in [*EC_MARKERS, "Other"] if s in set(subtype)]
    endo.obs = pd.DataFrame({
        "EC subtype": pd.Categorical(subtype, categories=order),
        "Leiden cluster": clusters.to_numpy(),
        "Harmonized label": endo.obs["Harmonized label"].cat.remove_unused_categories(),
        "Source dataset": endo.obs["Source dataset"].cat.remove_unused_categories(),
    }, index=endo.obs_names)
    endo.obsm["X_umap"] = tmp.obsm["X_umap"].astype(np.float32)

    print("EC subtype counts:", endo.obs["EC subtype"].value_counts().to_dict())
    print(pd.crosstab(endo.obs["Harmonized label"], endo.obs["EC subtype"]).to_string())
    if args.reference_labels.exists():
        ref = ad.read_h5ad(args.reference_labels, backed="r").obs["endo_group"].astype(str)
        shared = endo.obs_names.intersection(ref.index)
        print(f"concordance with reference endo_group on {len(shared):,} shared cells:")
        tab = pd.crosstab(ref[shared].rename("reference"), endo.obs.loc[shared, "EC subtype"].rename("new"))
        print(tab.to_string())
        agree = (ref[shared].to_numpy() == endo.obs.loc[shared, "EC subtype"].astype(str).to_numpy()).mean()
        print(f"agreement: {agree:.1%}")
    write(endo, "dolgalev_endothelial")


if __name__ == "__main__":
    main()
