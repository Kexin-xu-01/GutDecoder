from argparse import Namespace
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from loguru import logger

from gutdecoder.bench.benchmark import BenchmarkConfig, _fit_split, _parse_args, _resolve_args, predict_single_split


@pytest.fixture
def warnings_log():
    messages = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler)


def test_fit_split_passes_alpha_to_ridge():
    rng = np.random.default_rng(0)
    X_train, X_test = rng.normal(size=(40, 8)), rng.normal(size=(10, 8))
    y_train, y_test = rng.normal(size=(40, 2)), rng.normal(size=(10, 2))
    args = SimpleNamespace(dimreduce=None, seed=0, method="ridge", alpha=7.0)

    _, _, bundle, _, _ = _fit_split(X_train, X_test, y_train, y_test, args, genes=["a", "b"])

    assert bundle["regression_model"].alpha == 7.0


def test_resolve_args_order():
    cli = Namespace(seed=1, method="ridge", alpha=None)
    args = _resolve_args(cli, {"method": "xgboost"})

    assert args.seed == 1
    assert args.method == "xgboost"
    assert args.alpha == BenchmarkConfig().alpha


def test_resolve_args_config_wins_and_warns(tmp_path, warnings_log):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 5\nmethod: ridge\nnot_a_setting: 1\n")
    cli = Namespace(seed=5, method="xgboost", config=str(config))

    args = _resolve_args(cli, {})

    assert args.method == "ridge"
    assert len(warnings_log) == 2
    assert "method='xgboost' was passed via CLI" in warnings_log[0]
    assert "not_a_setting" in warnings_log[1]


def test_custom_encoder_does_not_turn_on_overwrite_for_later_runs(tmp_path):
    for name in ("train.csv", "test.csv"):
        (tmp_path / name).write_text("sample_id,patches_path,expr_path\n")
    args = Namespace(**asdict(BenchmarkConfig()))
    args.embed_dataroot = str(tmp_path / "embeddings")

    # No gene list exists, so the run stops right after the embedding step.
    with pytest.raises(FileNotFoundError):
        predict_single_split(
            str(tmp_path / "train.csv"), str(tmp_path / "test.csv"), args,
            str(tmp_path / "out"), "ds", "custom_encoder", device="cpu",
            bench_data_root=str(tmp_path), custom_encoder=None, extract_tiles=False,
        )

    assert args.overwrite is False


@pytest.mark.parametrize("argv, expected", [
    ([], None),
    (["--log_normalize", "true"], True),
    (["--log_normalize", "false"], False),
    # Old name and old spelling still work, and "False" now really means False.
    (["--normalize", "False"], False),
    (["--normalize", "True"], True),
])
def test_log_normalize_flag(argv, expected):
    assert _parse_args(argv).normalize is expected


@pytest.mark.parametrize("value, expected", [("true", True), ("false", False), ("0", False)])
def test_library_size_normalize_flag(value, expected):
    assert _parse_args(["--library_size_normalize", value]).library_size_normalize is expected


def test_normalize_flag_rejects_typos():
    with pytest.raises(SystemExit):
        _parse_args(["--log_normalize", "flase"])


@pytest.mark.parametrize("text, expected", [
    ("log_normalize: false\n", False),
    ("log_normalize: true\n", True),
    ("log_normalize: 'false'\n", False),
    ("normalize: false\n", False),  # old key still works
])
def test_config_file_accepts_log_normalize(tmp_path, text, expected):
    config = tmp_path / "config.yaml"
    config.write_text(text + "library_size_normalize: 'no'\n")

    args = _resolve_args(Namespace(config=str(config)), {})

    assert args.normalize is expected
    assert args.library_size_normalize is False


def test_config_file_rejects_conflicting_normalize_keys(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("log_normalize: true\nnormalize: false\n")

    with pytest.raises(ValueError, match="both log_normalize"):
        _resolve_args(Namespace(config=str(config)), {})
