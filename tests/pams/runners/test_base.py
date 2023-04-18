import json
import os.path
import random
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

import pytest

from pams import Simulator
from pams.logs import Logger
from pams.runners import Runner


class DummyRunner(Runner):
    def _setup(self) -> None:
        pass

    def _run(self) -> None:
        pass


class TestRunner:
    runner_class: Type[Runner] = DummyRunner
    default_setting: Dict = {"test": None}

    @pytest.mark.parametrize(
        "setting_mode", ["file_pointer", "file_path", "io", "dict"]
    )
    @pytest.mark.parametrize("logger", [None, Logger()])
    @pytest.mark.parametrize("simulator_class", [None, Simulator])
    def test__init__(
        self,
        setting_mode: str,
        logger: Optional[Logger],
        simulator_class: Optional[Type[Simulator]],
        setting: Optional[Dict] = None,
    ) -> Runner:
        if setting is None:
            setting = self.default_setting.copy()
        _prng = random.Random(42)
        kwargs: Dict[str, Any] = dict(prng=_prng, logger=logger)
        if simulator_class is not None:
            kwargs["simulator_class"] = simulator_class
        if setting_mode in ["file_pointer", "file_path"]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                setting_file = os.path.join(tmp_dir, "setting.json")
                json.dump(setting, open(setting_file, mode="w", encoding="utf-8"))
                if setting_mode == "file_pointer":
                    with open(setting_file, mode="r", encoding="utf-8") as fp:
                        runner = self.runner_class(settings=fp, **kwargs)
                elif setting_mode == "file_path":
                    runner = self.runner_class(settings=setting_file, **kwargs)
                else:
                    raise AssertionError
        elif setting_mode == "io":
            buffer = StringIO(json.dumps(setting))
            runner = self.runner_class(settings=buffer, **kwargs)
        elif setting_mode == "dict":
            runner = self.runner_class(settings=setting, **kwargs)
        else:
            raise AssertionError
        assert runner.settings == setting
        assert runner._prng == _prng
        assert runner.logger == logger
        assert isinstance(
            runner.simulator,
            simulator_class if simulator_class is not None else Simulator,
        )
        assert runner.registered_classes == []
        return runner

    def test_main(self) -> None:
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None
        )
        io = StringIO()
        with redirect_stdout(io):
            runner.main()
        print_values = io.getvalue().split("\n")
        assert print_values[-3].startswith("# INITIALIZATION TIME ")
        assert print_values[-2].startswith("# EXECUTION TIME ")

    def test_class_register(self) -> None:
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None
        )
        runner.class_register(cls=DummyRunner)
        assert DummyRunner in runner.registered_classes
