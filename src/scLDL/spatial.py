from __future__ import annotations

import numpy as np
import pandas as pd


RCTD_HIPPO = {
    "A": "Astrocyte",
    "CA1Pc": "CA1",
    "CA3Pc": "CA3",
    "DPc": "DG",
    "I": "Interneuron",
    "O": "Oligodendrocyte",
    "Pl": "OPC",
    "Es": "Endothelial",
    "Et": "Endothelial",
    "Ec": "Entorhinal",
    "E": "Ependymal",
    "MM": "Microglia",
    "Mr": "Mural",
    "NS": "Neurogenesis",
    "C": "Choroid",
    "CR": "Cajal_Retzius",
    "Nr": "Other_neuron",
}

BRAIN_MARKERS = {
    "Astrocyte": ["GJA1", "AQP4", "SLC1A2", "ALDH1L1"],
    "Astro": ["GJA1", "AQP4", "SLC1A2"],
    "Oligodendrocyte": ["MBP", "PLP1", "MOG", "MAG"],
    "Oligo": ["MBP", "PLP1", "MOG"],
    "OPC": ["PDGFRA", "CSPG4", "TNR"],
    "Microglia": ["C1QA", "C1QB", "CX3CR1", "TMEM119"],
    "Endothelial": ["FLT1", "CLDN5", "PECAM1"],
    "Mural": ["RGS5", "PDGFRB", "ACTA2"],
    "Ependymal": ["FOXJ1", "CFAP54"],
    "Interneuron": ["GAD1", "GAD2", "SLC32A1"],
    "DG": ["PROX1", "DOCK10"],
    "CA1": ["FIBCD1", "WFS1"],
    "CA2": ["AMIGO2", "RGS14"],
    "CA3": ["NECTIN3", "PVRL3", "BOK"],
    "CA": ["FIBCD1", "NECTIN3", "WFS1"],
    "Subiculum": ["DNER", "TPBG"],
    "SUB": ["DNER", "TPBG"],
    "Entorhinal": ["SLC17A6", "NELL1"],
    "Neurogenesis": ["DCX", "TOP2A"],
    "L2/3 IT CTX": ["CUX2", "SATB2"],
    "L4 IT CTX": ["RORB"],
    "L4/5 IT CTX": ["RORB", "RPRM"],
    "L5 IT CTX": ["DEPTOR"],
    "L5 PT CTX": ["BCL6", "FAM84B"],
    "L5/6 IT CTX": ["DEPTOR"],
    "L6 CT CTX": ["FOXP2", "SULF1"],
    "L6b CTX": ["CCN2", "CTGF"],
}


def standardize_gene_names(adata, copy: bool = True):
    """Uppercase gene symbols and drop duplicate columns (keep first)."""
    ad = adata.copy() if copy else adata
    names = pd.Index(ad.var_names.astype(str)).str.upper()
    ad.var_names = names
    if not ad.var_names.is_unique:
        _, idx = np.unique(ad.var_names.to_numpy(), return_index=True)
        ad = ad[:, np.sort(idx)].copy()
    return ad


def has_spatial(adata):
    if "spatial" in getattr(adata, "obsm", {}):
        return True
    return any(xk in adata.obs and yk in adata.obs for xk, yk in (("xcoord", "ycoord"), ("x", "y"), ("array_col", "array_row")))


def spatial_xy(adata):
    x, y = spatial_coords(adata)
    return np.column_stack([x, y])


def try_spatial_xy(adata):
    try:
        return spatial_xy(adata)
    except (KeyError, ValueError):
        return None


def spatial_coords(adata):
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"])[:, :2]
        return xy[:, 0], xy[:, 1]
    for xk, yk in (("xcoord", "ycoord"), ("x", "y"), ("array_col", "array_row")):
        if xk in adata.obs and yk in adata.obs:
            return adata.obs[xk].to_numpy(dtype=float), adata.obs[yk].to_numpy(dtype=float)
    raise KeyError("No spatial coordinates found (obsm['spatial'] or x/y obs columns).")


def subsample_balanced(adata, label_key: str, max_per_class: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(adata.obs[label_key].astype(str))
    keep = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return adata[keep].copy()


def collapse_dropviz(adata, min_cells: int = 40):
    """Map DropViz hippocampus metadata onto major anatomical types."""
    ad = adata.copy()
    cls = ad.obs["class"].astype(str).str.upper()
    name = ad.obs["common_name"].astype(str)
    out = []
    for c, n in zip(cls, name):
        nl = n.lower()
        if c in {"NAN", "NONE"} and n.lower() in {"nan", "none"}:
            out.append(None)
            continue
        if "dentate principal" in nl:
            out.append("DG")
        elif "ca1" in nl:
            out.append("CA1")
        elif "ca2" in nl:
            out.append("CA2")
        elif "ca3" in nl:
            out.append("CA3")
        elif "subiculum" in nl:
            out.append("Subiculum")
        elif "entorh" in nl:
            out.append("Entorhinal")
        elif "interneuron" in nl or "gaba" in nl:
            out.append("Interneuron")
        elif c == "ASTROCYTE" or "astrocyte" in nl:
            out.append("Astrocyte")
        elif c == "OLIGODENDROCYTE" or "oligo" in nl:
            out.append("Oligodendrocyte")
        elif c == "POLYDENDROCYTE" or "polydend" in nl:
            out.append("OPC")
        elif "endoth" in nl or c.startswith("ENDOTHELIAL"):
            out.append("Endothelial")
        elif c in {"MICROGLIA", "MACROPHAGE"} or "microglia" in nl:
            out.append("Microglia")
        elif c == "MURAL" or "mural" in nl:
            out.append("Mural")
        elif c == "EPENDYMAL" or "ependymal" in nl:
            out.append("Ependymal")
        elif c == "NEUROGENESIS" or "neurogenesis" in nl:
            out.append("Neurogenesis")
        elif c == "NEURON":
            out.append("Other_neuron")
        else:
            out.append(None)
    ad.obs["cell_type"] = out
    ad = ad[ad.obs["cell_type"].notna()].copy()
    counts = ad.obs["cell_type"].value_counts()
    keep = counts[counts >= min_cells].index
    return ad[ad.obs["cell_type"].isin(keep)].copy()


def map_rctd(series):
    return pd.Series(series).astype(str).map(lambda x: RCTD_HIPPO.get(x, x))


def marker_subset(classes, gene_names):
    genes = set(np.asarray(gene_names).astype(str))
    out = {}
    for lab in classes:
        cands = BRAIN_MARKERS.get(str(lab), [])
        hit = [g for g in cands if g in genes]
        if hit:
            out[str(lab)] = hit
    return out
