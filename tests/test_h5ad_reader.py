from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest

from gutdecoder.reader.h5ad_reader import rule_based_spot_exclusion


def _st(wsi):
    adata = ad.AnnData(np.zeros((3, 1)), obsm={"spatial": np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])})
    return SimpleNamespace(adata=adata, wsi=wsi)


@pytest.mark.parametrize("wsi", [None, SimpleNamespace(width=10, height=10)])
def test_rule_exclusion_without_rules_keeps_every_spot(wsi):
    keep, fig = rule_based_spot_exclusion(_st(wsi), [], plot=False, verbose=False)

    assert keep.tolist() == [True, True, True]
    assert fig is None


def test_rule_exclusion_does_not_hide_plot_errors():
    # No "total_counts" to colour by: used to print "Plotting failed" and return fig=None.
    with pytest.raises((KeyError, ValueError)):
        rule_based_spot_exclusion(_st(None), [], plot=True, verbose=False)
