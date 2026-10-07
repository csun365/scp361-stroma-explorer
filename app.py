import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
import scipy.sparse as sp
import streamlit as st

from datasets import DATASETS

st.set_page_config(page_title="Bone Marrow Niche Explorer", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#b4b2ad"
SEQ = [[0, "#cde2fb"], [0.35, "#6da7ec"], [0.7, "#256abf"], [1, "#0d366b"]]
DIVERGING = [[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]]
UP, DOWN = "#e34948", "#2a78d6"
PSEUDOCOUNT = 0.01  # on linear-scale means; keeps fold changes finite for genes absent in one group


def dataset_files(name):
    base = DATASETS[name]["file"]
    single = DATA_DIR / f"{base}.h5ad"
    if single.exists():
        return [single]
    parts = DATA_DIR.glob(f"{base}.part*of*.h5ad")
    return sorted(parts, key=lambda p: int(re.search(r"\.part(\d+)of", p.name).group(1)))


@st.cache_resource(max_entries=1, show_spinner="Loading dataset…")
def load(name):
    parts = [sc.read_h5ad(p) for p in dataset_files(name)]
    X = parts[0].X if len(parts) == 1 else sp.vstack([p.X for p in parts], format="csr")
    obs = pd.concat([p.obs for p in parts])
    obsm = {k: np.concatenate([p.obsm[k] for p in parts]) for k in parts[0].obsm}
    var = parts[0].var
    del parts
    # Kept row-compressed: converting the 32k-cell atlas to CSC would add ~0.7 GB at load.
    X = X.astype(np.float32, copy=False)
    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm)
    return {
        "adata": adata,
        "X": X,
        "genes": adata.var_names.to_numpy(),
        "gene_lookup": {g.lower(): g for g in adata.var_names},
        "gene_index": {g: i for i, g in enumerate(adata.var_names)},
    }


@st.cache_resource(max_entries=8, show_spinner="Summarizing groups…")
def group_stats(name, groupby):
    """Per-group sums of log and linear expression and detection counts (groups x genes)."""
    d = load(name)
    labels = d["adata"].obs[groupby]
    groups = list(labels.cat.categories)
    G = sp.csr_matrix(
        (np.ones(len(labels), dtype=np.float32), (labels.cat.codes.to_numpy(), np.arange(len(labels)))),
        shape=(len(groups), len(labels)),
    )
    X = d["X"]
    same_pattern = lambda data: sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
    as_df = lambda m: pd.DataFrame(np.asarray(m.todense()), index=groups, columns=d["genes"])
    return {
        "n": pd.Series(np.asarray(G.sum(1)).ravel(), index=groups),
        "sum_log": as_df(G @ X),
        "sum_lin": as_df(G @ same_pattern(np.expm1(X.data))),
        "n_detected": as_df(G @ same_pattern(np.ones_like(X.data))),
    }


@st.cache_data(max_entries=20, show_spinner=False)
def run_de(name, groupby, group_a, group_b, method):
    d = load(name)
    labels = d["adata"].obs[groupby]
    in_a, in_b = labels.isin(group_a).to_numpy(), labels.isin(group_b).to_numpy()
    keep = in_a | in_b
    sub = ad.AnnData(
        X=d["X"] if keep.all() else d["X"][keep],
        obs=pd.DataFrame({"g": pd.Categorical(np.where(in_a[keep], "A", "B"))}),
        var=pd.DataFrame(index=d["genes"]),
    )
    kwargs = {"tie_correct": True} if method == "wilcoxon" else {}
    sc.tl.rank_genes_groups(sub, "g", groups=["A"], reference="B", method=method, **kwargs)
    res = sc.get.rank_genes_groups_df(sub, "A").set_index("names").reindex(d["genes"])

    s = group_stats(name, groupby)
    a, b = list(group_a), list(group_b)
    n_a, n_b = s["n"][a].sum(), s["n"][b].sum()
    mean_a = s["sum_lin"].loc[a].sum() / n_a
    mean_b = s["sum_lin"].loc[b].sum() / n_b
    per_group_b = s["sum_lin"].loc[b].div(s["n"][b], axis=0)
    lfc = np.log2((mean_a + PSEUDOCOUNT) / (mean_b + PSEUDOCOUNT))
    margin_up = np.log2((mean_a + PSEUDOCOUNT) / (per_group_b.max() + PSEUDOCOUNT))
    margin_down = np.log2((per_group_b.min() + PSEUDOCOUNT) / (mean_a + PSEUDOCOUNT))

    out = pd.DataFrame({
        "score": res["scores"],
        "log2FC": lfc,
        "padj": res["pvals_adj"],
        "pct_A": s["n_detected"].loc[a].sum() / n_a * 100,
        "pct_B": s["n_detected"].loc[b].sum() / n_b * 100,
        "mean_A": mean_a,
        "mean_B": mean_b,
        "specificity": margin_up.where(lfc >= 0, margin_down),
    })
    out.index.name = "gene"
    return out, int(n_a), int(n_b)


def parse_genes(name, text):
    lookup = load(name)["gene_lookup"]
    tokens = [t for t in text.replace(",", " ").replace(";", " ").split() if t]
    found = [lookup[t.lower()] for t in tokens if t.lower() in lookup]
    missing = [t for t in tokens if t.lower() not in lookup]
    return list(dict.fromkeys(found)), missing


def gene_vector(name, gene):
    d = load(name)
    return d["X"][:, d["gene_index"][gene]].toarray().ravel()


def base_embedding_fig(title, height):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        legend=dict(itemsizing="constant"),
        hoverlabel=dict(namelength=-1),
    )
    return fig


def cluster_fig(xy, labels, highlight, point_size, height=520):
    fig = base_embedding_fig(f"{labels.name}s", height)
    rest = ~labels.isin(highlight).to_numpy()
    fig.add_trace(go.Scattergl(
        x=xy[rest, 0], y=xy[rest, 1], mode="markers", name="other",
        marker=dict(size=point_size, color=MUTED, opacity=0.5),
        text=labels[rest], hovertemplate=f"{labels.name} %{{text}}<extra></extra>",
    ))
    for color, grp in zip(SERIES, highlight):
        m = (labels == grp).to_numpy()
        fig.add_trace(go.Scattergl(
            x=xy[m, 0], y=xy[m, 1], mode="markers", name=str(grp),
            marker=dict(size=point_size, color=color, opacity=0.85),
            hovertemplate=f"{labels.name} {grp}<extra></extra>",
        ))
    centroids = pd.DataFrame(xy, columns=["x", "y"]).groupby(labels.to_numpy(), observed=True).median()
    fig.add_trace(go.Scatter(
        x=centroids["x"], y=centroids["y"], mode="text", text=centroids.index,
        textfont=dict(size=12), hoverinfo="skip", showlegend=False,
    ))
    return fig


def gene_fig(name, xy, gene, labels, point_size, height=420):
    vals = gene_vector(name, gene)
    fig = base_embedding_fig(gene, height)
    zero = vals == 0
    fig.add_trace(go.Scattergl(
        x=xy[zero, 0], y=xy[zero, 1], mode="markers", name="not detected",
        marker=dict(size=point_size, color=MUTED, opacity=0.35), hoverinfo="skip",
    ))
    order = np.where(~zero)[0]
    order = order[np.argsort(vals[order])]
    vmax = float(np.percentile(vals[order], 99)) if len(order) else 1.0
    fig.add_trace(go.Scattergl(
        x=xy[order, 0], y=xy[order, 1], mode="markers", name="detected",
        marker=dict(size=point_size, color=vals[order], colorscale=SEQ, cmin=0, cmax=max(vmax, 1e-3),
                    colorbar=dict(title="log expr", thickness=10, len=0.7)),
        customdata=np.stack([labels.to_numpy()[order].astype(str), vals[order]], axis=1),
        hovertemplate=f"{labels.name} %{{customdata[0]}}<br>{gene}: %{{customdata[1]:.2f}}<extra></extra>",
    ))
    fig.update_layout(showlegend=False)
    pct = 100 * (~zero).mean()
    fig.add_annotation(text=f"detected in {pct:.1f}% of cells", xref="paper", yref="paper",
                       x=0.01, y=0.0, showarrow=False, font=dict(size=11))
    return fig


def dotplot_fig(name, genes, groupby, mark_a=(), mark_b=()):
    s = group_stats(name, groupby)
    groups = list(s["n"].index)
    mean_log = s["sum_log"][genes].div(s["n"], axis=0)
    pct = s["n_detected"][genes].div(s["n"], axis=0) * 100
    scaled = (mean_log - mean_log.min()) / (mean_log.max() - mean_log.min()).replace(0, 1)
    tag = lambda g: f"{g} (A)" if g in mark_a else (f"{g} (B)" if g in mark_b else str(g))
    xlabels = [tag(g) for g in groups]
    gx, gy = np.meshgrid(xlabels, genes)
    sizes = pct.T.to_numpy()
    fig = go.Figure(go.Scatter(
        x=gx.ravel(), y=gy.ravel(), mode="markers",
        marker=dict(size=3 + sizes.ravel() * 0.22, color=scaled.T.to_numpy().ravel(), colorscale=SEQ,
                    cmin=0, cmax=1, line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
                    colorbar=dict(title="scaled<br>mean", thickness=10, len=0.6)),
        customdata=np.stack([mean_log.T.to_numpy().ravel(), sizes.ravel()], axis=1),
        hovertemplate="%{y} in %{x}<br>mean log expr %{customdata[0]:.2f}<br>"
                      "detected %{customdata[1]:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(360, 22 * len(genes) + 160), margin=dict(l=10, r=10, t=30, b=10),
        dragmode=False,
        xaxis=dict(type="category", tickmode="array", tickvals=xlabels, ticktext=xlabels, tickangle=-60,
                   showgrid=False, categoryorder="array", categoryarray=xlabels, fixedrange=True),
        yaxis=dict(type="category", tickmode="array", tickvals=genes, ticktext=genes, autorange="reversed",
                   showgrid=False, categoryorder="array", categoryarray=genes, fixedrange=True),
        title=dict(text="Dot size = % of cells detected · color = mean expression scaled per gene",
                   x=0.01, font=dict(size=13)),
    )
    return fig


def heatmap_fig(name, genes, groupby, mark_a=(), mark_b=()):
    s = group_stats(name, groupby)
    mean_log = s["sum_log"][genes].div(s["n"], axis=0)
    z = (mean_log - mean_log.mean()) / mean_log.std().replace(0, 1)
    tag = lambda g: f"{g} (A)" if g in mark_a else (f"{g} (B)" if g in mark_b else str(g))
    xlabels = [tag(g) for g in z.index]
    fig = go.Figure(go.Heatmap(
        z=z.T.to_numpy(), x=xlabels, y=genes, colorscale=DIVERGING, zmid=0,
        xgap=1, ygap=1, colorbar=dict(title="z-score", thickness=10, len=0.6),
        hovertemplate="%{y} in %{x}<br>z = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(360, 22 * len(genes) + 160), margin=dict(l=10, r=10, t=30, b=10),
        dragmode=False,
        yaxis=dict(type="category", tickmode="array", tickvals=genes, ticktext=genes,
                   autorange="reversed", fixedrange=True),
        xaxis=dict(type="category", tickmode="array", tickvals=xlabels, ticktext=xlabels,
                   tickangle=-60, fixedrange=True),
        title=dict(text="Group mean expression, z-scored per gene", x=0.01, font=dict(size=13)),
    )
    return fig


def volcano_fig(df, up, down, padj_max, lfc_min, label_genes):
    y = -np.log10(df["padj"].clip(lower=1e-300))
    fig = go.Figure()
    for name, mask, color, opacity in [
        ("not selected", ~(up | down), MUTED, 0.4),
        ("down in A", down, DOWN, 0.75),
        ("up in A", up, UP, 0.75),
    ]:
        sub = df[mask]
        fig.add_trace(go.Scattergl(
            x=sub["log2FC"], y=y[mask], mode="markers", name=f"{name} ({mask.sum()})",
            marker=dict(size=6, color=color, opacity=opacity),
            text=sub.index,
            customdata=np.stack([sub["padj"], sub["pct_A"], sub["pct_B"]], axis=1) if len(sub) else None,
            hovertemplate="<b>%{text}</b><br>log2FC %{x:.2f}<br>padj %{customdata[0]:.2e}"
                          "<br>%A %{customdata[1]:.1f} · %B %{customdata[2]:.1f}<extra></extra>",
        ))
    for g in label_genes:
        fig.add_annotation(x=df.at[g, "log2FC"], y=y[g], text=g, showarrow=True, arrowhead=0,
                           arrowwidth=1, ax=20, ay=-18, font=dict(size=11))
    fig.add_hline(y=-np.log10(padj_max), line=dict(dash="dot", width=1, color=MUTED))
    for x in (-lfc_min, lfc_min):
        fig.add_vline(x=x, line=dict(dash="dot", width=1, color=MUTED))
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="log2 fold change (A vs B)", yaxis_title="−log10 adjusted p",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


# ---------------------------------------------------------------- layout
with st.sidebar:
    st.header("Settings")
    ds = st.selectbox("Dataset", list(DATASETS), key="dataset")
    spec = DATASETS[ds]
    if st.session_state.get("_loaded") not in (None, ds):
        load.clear()  # drop the previous dataset before loading the next, so both never sit in memory
    st.session_state["_loaded"] = ds
    d = load(ds)
    adata = d["adata"]
    embedding = st.radio("Embedding", list(spec["embeddings"]), key=f"emb_{ds}")
    groupby = st.radio("Group cells by", spec["groupings"], key=f"grp_{ds}", help=spec.get("grouping_help"))
    point_size = st.slider("Point size", 1, 6, 3)
    st.caption(f"{adata.n_obs:,} cells · {adata.n_vars:,} genes (detected in ≥3 cells)")

xy = adata.obsm[spec["embeddings"][embedding]]
labels = adata.obs[groupby]
all_groups = list(labels.cat.categories)
units = spec["units"]

st.title(spec["title"])
tab_genes, tab_de, tab_about = st.tabs(["Gene explorer", "Differential expression", "About"])

with tab_genes:
    c1, c2 = st.columns([2, 1])
    with c1:
        picked = st.multiselect("Genes", d["genes"], key=f"genes_{ds}",
                                default=[g for g in spec["default_genes"] if g in d["gene_index"]],
                                placeholder="Type to search…")
    with c2:
        pasted = st.text_input("…or paste a list", placeholder="Fgf7, Fgf18, Ntn1", key=f"paste_{ds}")
    extra, missing = parse_genes(ds, pasted)
    if missing:
        st.warning(f"Not found (or detected in <3 cells): {', '.join(missing)}")
    genes = list(dict.fromkeys(picked + extra))

    highlight = st.multiselect(
        f"Highlight up to {len(SERIES)} groups (others shown in gray, all are labeled)",
        all_groups, default=all_groups[: len(SERIES)], max_selections=len(SERIES), key=f"hl_{ds}_{groupby}",
    )
    st.plotly_chart(cluster_fig(xy, labels, highlight, point_size), width="stretch", key="explorer_groups")

    if genes:
        cols = st.columns(2)
        for i, g in enumerate(genes):
            with cols[i % 2]:
                st.plotly_chart(gene_fig(ds, xy, g, labels, point_size), width="stretch", key=f"explorer_gene_{g}")
        st.subheader(f"Expression by {groupby.lower()}")
        st.plotly_chart(dotplot_fig(ds, genes, groupby), width="stretch", key="explorer_dotplot")
    else:
        st.info("Pick or paste genes to plot their expression.")

with tab_de:
    st.markdown(f"Compare group **A** against group **B** (each a set of {groupby.lower()} groups).")
    default_a = spec.get("default_group_a", {}).get(groupby)
    with st.form(f"de_form_{ds}_{groupby}"):
        c1, c2 = st.columns(2)
        with c1:
            group_a = st.multiselect("Group A", all_groups, key=f"ga_{ds}_{groupby}",
                                     default=[default_a] if default_a in all_groups else all_groups[:1])
        with c2:
            b_mode = st.radio("Group B", ["All other cells", "Choose groups"], horizontal=True, key=f"bm_{ds}_{groupby}")
            group_b_pick = st.multiselect("Group B groups (used when 'Choose groups' is selected)", all_groups,
                                          key=f"gb_{ds}_{groupby}")
        method = st.selectbox("Test", ["wilcoxon", "t-test"], key=f"test_{ds}_{groupby}",
                              help="Wilcoxon rank-sum is the standard for scRNA-seq; t-test is faster.")
        submitted = st.form_submit_button("Run differential expression", type="primary")

    if submitted:
        group_b = [g for g in all_groups if g not in group_a] if b_mode == "All other cells" else group_b_pick
        overlap = set(group_a) & set(group_b)
        if not group_a or not group_b:
            st.error("Both groups need at least one group selected.")
        elif overlap:
            st.error(f"Groups overlap: {', '.join(sorted(overlap))}")
        else:
            st.session_state[f"de_args_{ds}"] = (ds, groupby, tuple(group_a), tuple(group_b), method)

    args = st.session_state.get(f"de_args_{ds}")
    if args:
        _, de_groupby, ga, gb, de_method = args
        with st.spinner("Running test across all genes… (Wilcoxon on large groups can take ~30–60 s)"):
            df, n_a, n_b = run_de(*args)
        st.success(f"{de_method} · A = {de_groupby} {', '.join(ga)} ({n_a:,} cells) vs "
                   f"B = {de_groupby} {', '.join(gb) if len(gb) <= 10 else f'{len(gb)} groups'} ({n_b:,} cells)")

        rank_options = {
            "Test score": "score",
            "log2 fold change": "log2FC",
            "Detection difference (%A − %B)": "pct_diff",
            "Specificity margin (vs closest B group)": "specificity",
        }
        f1, f2, f3 = st.columns(3)
        with f1:
            direction = st.radio("Direction", ["Up in A", "Down in A", "Both"], horizontal=True)
            rank_label = st.selectbox(
                "Rank by", list(rank_options),
                help="Specificity margin: log2 ratio of A's mean to the highest-expressing individual "
                     "B group (or lowest, for down genes). Positive = A beats every B group, "
                     "not just the B average.",
            )
        with f2:
            padj_max = st.number_input("Max adjusted p", 0.0, 1.0, 0.05, format="%.3g")
            lfc_min = st.number_input("Min |log2FC|", 0.0, 20.0, 1.0, step=0.25)
        with f3:
            min_pct_a = st.slider("Min % of cells detected in the higher group", 0, 100, 10)
            max_pct_b = st.slider("Max % detected in the lower group", 0, 100, 100,
                                  help="Lower this to require 'on/off' markers.")

        df = df.assign(pct_diff=df["pct_A"] - df["pct_B"])
        sig = (df["padj"] <= padj_max) & (df["log2FC"].abs() >= lfc_min)
        hi = np.where(df["log2FC"] >= 0, df["pct_A"], df["pct_B"])
        lo = np.where(df["log2FC"] >= 0, df["pct_B"], df["pct_A"])
        sig &= (hi >= min_pct_a) & (lo <= max_pct_b)
        up = sig & (df["log2FC"] > 0)
        down = sig & (df["log2FC"] < 0)
        selected = {"Up in A": up, "Down in A": down, "Both": up | down}[direction]

        key = rank_options[rank_label]
        sign = np.where(df["log2FC"] >= 0, 1, -1)
        rank_value = df[key] if key == "specificity" else df[key] * (sign if direction == "Both" else (1 if direction == "Up in A" else -1))
        table = df[selected].assign(_rank=rank_value[selected]).sort_values("_rank", ascending=False).drop(columns="_rank")

        m1, m2, m3 = st.columns(3)
        m1.metric("Up in A", f"{int(up.sum()):,}")
        m2.metric("Down in A", f"{int(down.sum()):,}")
        m3.metric("Genes tested", f"{len(df):,}")

        top_n = st.slider("Genes to show in plots", 5, 50, 20)
        top = list(table.index[:top_n])

        v1, v2 = st.columns([3, 2])
        with v1:
            st.plotly_chart(volcano_fig(df, up, down, padj_max, lfc_min, top[:10]), width="stretch", key="de_volcano")
        with v2:
            st.markdown(f"**Top genes** · ranked by {rank_label.lower()}")
            st.dataframe(
                table.drop(columns=["pct_diff"]).reset_index(),
                height=470, hide_index=True,
                column_config={
                    "score": st.column_config.NumberColumn(format="%.1f"),
                    "log2FC": st.column_config.NumberColumn(format="%.2f"),
                    "padj": st.column_config.NumberColumn(format="%.1e"),
                    "pct_A": st.column_config.NumberColumn("% A", format="%.1f"),
                    "pct_B": st.column_config.NumberColumn("% B", format="%.1f"),
                    "mean_A": st.column_config.NumberColumn(f"mean A ({units})", format="%.2f"),
                    "mean_B": st.column_config.NumberColumn(f"mean B ({units})", format="%.2f"),
                    "specificity": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            safe = lambda groups: re.sub(r"[^A-Za-z0-9_.-]+", "", "-".join(groups))
            st.download_button(
                "Download filtered table (CSV)", table.to_csv().encode(),
                file_name=f"DE_{ds.replace(' ', '_')}_{safe([de_groupby])}_{safe(ga)}_vs_"
                          f"{safe(gb) if len(gb) <= 6 else 'rest'}.csv",
                mime="text/csv",
            )

        if top:
            st.subheader("Top genes across all groups")
            p1, p2 = st.columns(2)
            with p1:
                st.plotly_chart(dotplot_fig(ds, top, de_groupby, ga, gb), width="stretch", key="de_dotplot")
            with p2:
                st.plotly_chart(heatmap_fig(ds, top, de_groupby, ga, gb), width="stretch", key="de_heatmap")

            st.subheader("View a result gene on the embedding")
            g = st.selectbox("Gene", list(table.index[:200]))
            e1, e2 = st.columns(2)
            with e1:
                st.plotly_chart(gene_fig(ds, xy, g, labels, point_size), width="stretch", key="de_gene_view")
            with e2:
                st.plotly_chart(cluster_fig(xy, adata.obs[de_groupby], list(ga)[: len(SERIES)], point_size, height=420),
                                width="stretch", key="de_groups_view")
        else:
            st.info("No genes pass the current filters.")
    else:
        st.info("Choose groups and press **Run differential expression**.")

with tab_about:
    st.markdown(spec["about"])
    st.markdown(f"""
**Statistics:** p-values come from scanpy's `rank_genes_groups` (Wilcoxon rank-sum with tie correction,
or Welch t-test) with Benjamini–Hochberg correction. Fold changes are computed from linear-scale ({units})
group means with a pseudocount of {PSEUDOCOUNT}, so genes absent from one group get large but finite values.

**Specificity margin:** for genes up in A, log2(mean A / mean of the highest single B group); for genes
down in A, log2(mean of the lowest single B group / mean A). A positive value means A differs from
*every* B group in that direction, not only from the pooled B average.

Note: with thousands of cells, almost any consistent difference is "significant"; rank by effect size or
specificity and use the detection filters to find useful markers.
""")
