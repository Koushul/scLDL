import numpy as np
from anndata import AnnData

from scLDL.spatial import (
    RCTD_HIPPO,
    collapse_dropviz,
    map_rctd,
    spatial_coords,
    standardize_gene_names,
    subsample_balanced,
)


def test_standardize_gene_names_upper_and_dedup():
    ad = AnnData(np.arange(6, dtype=np.float32).reshape(2, 3))
    ad.var_names = ["Slc17a7", "slc17a7", "Gad1"]
    out = standardize_gene_names(ad)
    assert list(out.var_names) == ["SLC17A7", "GAD1"]
    assert out.n_vars == 2


def test_spatial_coords_from_obs():
    ad = AnnData(np.ones((3, 2), dtype=np.float32))
    ad.obs["xcoord"] = [0, 1, 2]
    ad.obs["ycoord"] = [4, 5, 6]
    x, y = spatial_coords(ad)
    assert list(x) == [0, 1, 2]
    assert list(y) == [4, 5, 6]


def test_subsample_balanced():
    ad = AnnData(np.ones((10, 2), dtype=np.float32))
    ad.obs["cell_type"] = ["a"] * 7 + ["b"] * 3
    out = subsample_balanced(ad, "cell_type", max_per_class=2, seed=0)
    assert out.n_obs == 4
    assert out.obs["cell_type"].value_counts().max() == 2


def test_rctd_and_dropviz_maps():
    assert list(map_rctd(["CA1Pc", "A"])) == ["CA1", "Astrocyte"]
    ad = AnnData(np.ones((5, 2), dtype=np.float32))
    ad.obs["class"] = ["NEURON", "ASTROCYTE", "nan", "NEURON", "OLIGODENDROCYTE"]
    ad.obs["common_name"] = [
        "CA1 Principal cells",
        "Astrocyte.Gja1.Nnat",
        "nan",
        "Interneuron, Basket 1",
        "Oligodendrocyte.Trf.Il33",
    ]
    out = collapse_dropviz(ad, min_cells=1)
    assert set(out.obs["cell_type"]) == {"CA1", "Astrocyte", "Interneuron", "Oligodendrocyte"}
    assert "CA1Pc" in RCTD_HIPPO
