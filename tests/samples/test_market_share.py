import os.path
from unittest import mock

from samples.market_share.main import main


def test_market_share() -> None:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    sample_dir = os.path.join(root_dir, "samples", "market_share")
    with mock.patch(
        "sys.argv", ["main.py", "--config", f"{sample_dir}/config.json", "--seed", "1"]
    ):
        main()
    with mock.patch(
        "sys.argv",
        ["main.py", "--config", f"{sample_dir}/config-mm.json", "--seed", "1"],
    ):
        main()
