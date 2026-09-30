"""Load the sample configurations that the tutorials copy."""
import json
import os
from typing import Any
from typing import Dict

SAMPLES_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "samples"
)

# the old names of session keys, which some samples still use
RENAMED_KEYS: Dict[str, str] = {
    "hifreqSubmitRate": "highFrequencySubmitRate",
    "maxHifreqOrders": "maxHighFrequencyOrders",
}


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            RENAMED_KEYS.get(key, key): _clean(item)
            for key, item in value.items()
            if key != "MEMO"
        }
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def load_sample_config(sample: str, file: str = "config.json") -> Dict[str, Any]:
    """Read a sample configuration without its MEMO keys and with the new key names."""
    with open(os.path.join(SAMPLES_DIR, sample, file), encoding="utf-8") as fp:
        config: Dict[str, Any] = _clean(json.load(fp))
    return config
