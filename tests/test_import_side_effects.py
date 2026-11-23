import subprocess
import sys


def _run(code):
    # A fresh interpreter, so earlier imports in the test session cannot hide a side effect.
    subprocess.run([sys.executable, "-c", code], check=True)


def test_importing_benchmark_does_not_set_torch_sharing_strategy():
    # HEST sets the strategy itself when hest.bench is imported; only gutdecoder's own calls count.
    _run(
        "import inspect\n"
        "import torch.multiprocessing as mp\n"
        "callers, original = [], mp.set_sharing_strategy\n"
        "def spy(strategy):\n"
        "    callers.append(inspect.stack()[1].filename)\n"
        "    original(strategy)\n"
        "mp.set_sharing_strategy = spy\n"
        "import gutdecoder.bench.benchmark\n"
        "print('set_sharing_strategy called from:', callers)\n"
        "assert not [c for c in callers if 'site-packages/gutdecoder/' in c], callers\n"
    )


def test_importing_trainer_does_not_set_tqdm_env():
    _run(
        "import os\n"
        "os.environ.pop('TQDM_DISABLE', None)\n"
        "import gutdecoder.bench.trainer\n"
        "assert 'TQDM_DISABLE' not in os.environ\n"
    )


def test_importing_reader_keeps_scanpy_qc_function():
    _run(
        "import scanpy as sc\n"
        "before = sc.pp.calculate_qc_metrics\n"
        "import gutdecoder.reader.h5ad_reader\n"
        "assert sc.pp.calculate_qc_metrics is before\n"
    )
