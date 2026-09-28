import os.path
from unittest import mock

import pytest

from samples.fat_tail.main import main


def test_fat_tail(capsys: pytest.CaptureFixture[str]) -> None:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    sample_dir = os.path.join(root_dir, "samples", "fat_tail")
    with mock.patch(
        "sys.argv", ["main.py", "--config", f"{sample_dir}/config.json", "--seed", "1"]
    ):
        main()
    # one line per step with the columns of plhamJ: session, time, market ID,
    # market name, market price and fundamental price
    lines = [
        line.split()
        for line in capsys.readouterr().out.splitlines()
        if not line.startswith("#")
    ]
    assert [line[0] for line in lines] == ["0"] * 500 + ["1"] * 2000
    assert all(len(line) == 6 for line in lines)
    # the fixed margins keep the market price around the fundamental price of 300
    assert all(200.0 < float(line[4]) < 400.0 for line in lines)
