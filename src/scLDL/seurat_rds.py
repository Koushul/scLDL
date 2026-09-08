from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import sparse


def _tag_name(tag):
    from rdata.parser._parser import RObjectType

    if tag is None:
        return None
    if tag.info.type == RObjectType.REF:
        return _tag_name(tag.referenced_object)
    if tag.info.type == RObjectType.SYM:
        v = tag.value
        if hasattr(v, "value") and isinstance(v.value, (bytes, bytearray)):
            return v.value.decode("utf-8", "replace")
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", "replace")
    if tag.info.type == RObjectType.CHAR:
        v = tag.value
        return v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v)
    return None


def _pairlist(node):
    from rdata.parser._parser import RObjectType

    out = OrderedDict()
    while node is not None and node.info.type == RObjectType.LIST:
        out[_tag_name(node.tag)] = node.value[0]
        node = node.value[1]
    return out


def _as_str_array(obj):
    from rdata.parser._parser import RObjectType

    if obj.info.type == RObjectType.REF:
        return _as_str_array(obj.referenced_object)
    if obj.info.type == RObjectType.STR:
        out = []
        for o in obj.value:
            if o.info.type == RObjectType.CHAR:
                v = o.value
                out.append(v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v))
            elif o.info.type == RObjectType.REF:
                out.extend(_as_str_array(o.referenced_object).tolist())
            else:
                out.append("")
        return np.asarray(out, dtype=object)
    if obj.info.type == RObjectType.INT:
        return np.asarray(obj.value)
    raise TypeError(obj.info.type)


def _decode_vector(obj):
    from rdata.parser._parser import RObjectType

    t = obj.info.type
    attrs = _pairlist(obj.attributes) if obj.attributes else {}
    levels = attrs.get("levels")
    if t == RObjectType.INT and levels is not None:
        idx = np.asarray(obj.value)
        lev = _as_str_array(levels)
        out = np.array(["NA"] * len(idx), dtype=object)
        ok = (idx >= 1) & (idx <= len(lev))
        out[ok] = lev[idx[ok] - 1]
        return out
    if t == RObjectType.STR:
        return _as_str_array(obj)
    if t in (RObjectType.INT, RObjectType.REAL, RObjectType.LGL):
        return np.asarray(obj.value)
    raise TypeError(t)


def _as_matrix(obj):
    from rdata.parser._parser import RObjectType

    if obj.info.type == RObjectType.REF:
        return _as_matrix(obj.referenced_object)
    attrs = _pairlist(obj.attributes) if obj.attributes else {}
    dim = np.asarray(attrs["dim"].value, dtype=int)
    data = np.asarray(obj.value)
    return np.ascontiguousarray(data.reshape(dim, order="F"))


def _as_dgc(obj):
    from rdata.parser._parser import RObjectType

    if obj.info.type == RObjectType.REF:
        return _as_dgc(obj.referenced_object)
    sl = _pairlist(obj.attributes)
    i = np.asarray(sl["i"].value, dtype=np.int32)
    p = np.asarray(sl["p"].value, dtype=np.int32)
    x = np.asarray(sl["x"].value, dtype=np.float32)
    shape = tuple(int(v) for v in np.asarray(sl["Dim"].value))
    return sparse.csc_matrix((x, i, p), shape=shape)


def load_seurat_rds(path, assay: str = "RNA", layer: str = "counts"):
    import pandas as pd
    import rdata
    from rdata.parser._parser import RObjectType

    parsed = rdata.parser.parse_file(Path(path))
    seurat = _pairlist(parsed.object.attributes)

    meta_obj = seurat["meta.data"]
    colnames = _as_str_array(_pairlist(meta_obj.attributes)["names"])
    meta = {str(name): _decode_vector(meta_obj.value[i]) for i, name in enumerate(colnames)}
    obs = pd.DataFrame(meta)
    rownames = _pairlist(meta_obj.attributes).get("row.names")
    if rownames is not None and rownames.info.type == RObjectType.STR:
        obs.index = _as_str_array(rownames).astype(str)

    assays = seurat["assays"]
    assay_names = _as_str_array(_pairlist(assays.attributes)["names"])
    idx = list(assay_names).index(assay)
    assay_obj = assays.value[idx]
    layers = _pairlist(assay_obj.attributes)
    mat = _as_dgc(layers[layer])
    dn_obj = _pairlist(layers[layer].attributes)["Dimnames"]
    if dn_obj.info.type != RObjectType.VEC:
        raise TypeError("expected Dimnames list")
    genes = _as_str_array(dn_obj.value[0]).astype(str)
    cells = _as_str_array(dn_obj.value[1]).astype(str)
    obs = obs.reindex(cells)

    obsm = {}
    red = seurat.get("reductions")
    if red is not None:
        rnames = _as_str_array(_pairlist(red.attributes)["names"])
        for i, name in enumerate(rnames):
            rslots = _pairlist(red.value[i].attributes)
            if "cell.embeddings" in rslots:
                obsm[f"X_{name}"] = _as_matrix(rslots["cell.embeddings"])

    import anndata as ad

    adata = ad.AnnData(X=mat.T.tocsr(), obs=obs)
    adata.var_names = genes
    adata.obs_names = cells
    for k, v in obsm.items():
        adata.obsm[k] = v
    return adata
