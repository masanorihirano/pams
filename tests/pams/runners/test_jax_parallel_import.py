"""Tests of JaxAgentParallelRunner that do not require JAX, so they run whether it is installed."""
import copy
import os
import subprocess
import sys
from typing import Dict

import pytest

import pams
from pams.runners import JaxAgentParallelRunner

SETTING: Dict = {
    "simulation": {
        "markets": ["Market"],
        "agents": ["FCNAgents"],
        "sessions": [
            {
                "sessionName": 0,
                "iterationSteps": 10,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": True,
                "maxNormalOrders": 4,
            }
        ],
        "numParallel": 2,
    },
    "Market": {"class": "Market", "tickSize": 0.00001, "marketPrice": 300.0},
    "FCNAgents": {
        "class": "FCNAgent",
        "numAgents": 10,
        "markets": ["Market"],
        "assetVolume": 50,
        "cashAmount": 10000,
        "fundamentalWeight": {"expon": [1.0]},
        "chartWeight": {"expon": [0.0]},
        "noiseWeight": {"expon": [1.0]},
        "noiseScale": 0.001,
        "timeWindowSize": [100, 200],
        "orderMargin": [0.0, 0.1],
    },
}


def test_import_does_not_import_jax() -> None:
    code = (
        "import sys\n"
        "import pams\n"
        "from pams.runners import JaxAgentParallelRunner\n"
        "print([name for name in ['jax', 'flax'] if name in sys.modules])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(pams.__file__))),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


def test_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules makes `import jax` raise ImportError even if JAX is installed
    monkeypatch.setitem(sys.modules, "jax", None)
    with pytest.raises(ImportError, match="requires JAX") as exc_info:
        JaxAgentParallelRunner(settings=copy.deepcopy(SETTING))
    assert "pip install jax" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ImportError)
