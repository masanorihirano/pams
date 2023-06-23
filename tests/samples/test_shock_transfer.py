import os.path
from unittest import mock

from samples.shock_transfer.main import main


def test_shock_transfer() -> None:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    sample_dir = os.path.join(root_dir, "samples", "shock_transfer")
    with mock.patch(
        "sys.argv", ["main.py", "--config", f"{sample_dir}/config.json", "--seed", "1"]
    ):
        main()
