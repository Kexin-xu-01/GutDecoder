import numpy as np
import pytest

from gutdecoder.reader.crop_image import parse_crop_rules

RULE = {"type": "strip", "side": "left", "value": 0.1}


@pytest.mark.parametrize("cell", [
    "{'type': 'strip', 'side': 'left', 'value': 0.1}",
    '{"type": "strip", "side": "left", "value": 0.1}',
    "[{'type': 'strip', 'side': 'left', 'value': 0.1}]",
    '{""type"": ""strip"", ""side"": ""left"", ""value"": 0.1} junk',
])
def test_parses_one_rule_in_any_quoting(cell):
    assert parse_crop_rules(cell) == [RULE]


def test_parses_several_rules():
    cell = "{'type': 'strip', 'side': 'left', 'value': 0.1},\n{'type': 'strip', 'side': 'top', 'value': 5}"

    assert [r["side"] for r in parse_crop_rules(cell)] == ["left", "top"]


@pytest.mark.parametrize("cell", [None, np.nan, "  "])
def test_empty_cell_means_no_rules(cell):
    assert parse_crop_rules(cell) == []


@pytest.mark.parametrize("cell", ["left 10%", "{'type': 'strip', 'side': left}"])
def test_malformed_rule_raises_instead_of_leaving_slide_unmasked(cell):
    with pytest.raises(ValueError):
        parse_crop_rules(cell)
