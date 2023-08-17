import os.path
from unittest import mock

from samples.fat_finger.main import main


def test_fat_finger() -> None:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    sample_dir = os.path.join(root_dir, "samples", "fat_finger")
    with mock.patch(
        "sys.argv", ["main.py", "--config", f"{sample_dir}/config.json", "--seed", "1"]
    ):
        main()
