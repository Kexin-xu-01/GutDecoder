import pytest

from gutdecoder.utils.workflow_config import benchmark_arguments, coerce_bool


@pytest.mark.parametrize("value", [True, 1, "1", "true", "True", " yes ", "Y", "on"])
def test_coerce_bool_true(value):
    assert coerce_bool(value) is True


@pytest.mark.parametrize("value", [False, 0, None, "0", "false", "FALSE", "no", "n", "off", "none", ""])
def test_coerce_bool_false(value):
    assert coerce_bool(value) is False


@pytest.mark.parametrize("value", ["flase", "ture", "2", 5, "maybe"])
def test_coerce_bool_rejects_unclear_values(value):
    with pytest.raises(ValueError, match="Expected a boolean"):
        coerce_bool(value)


def test_benchmark_arguments_rejects_typo_in_symlink():
    with pytest.raises(ValueError, match="flase"):
        benchmark_arguments({"args": {"symlink": "flase"}})
