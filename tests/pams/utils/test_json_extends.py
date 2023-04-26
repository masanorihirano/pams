import pytest

from pams.utils import json_extends


def test_json_extends() -> None:
    whole_json = {"parent": {"parent_attr": "1", "override_attr": "1"}}
    target_json = {"extends": "parent", "child_attr": "0", "override_attr": "0"}

    result = json_extends(
        whole_json=whole_json, parent_name="child", target_json=target_json
    )
    expected = {"parent_attr": "1", "child_attr": "0", "override_attr": "0"}
    assert result == expected

    whole_json2 = {
        "grand_parent": {"grand_parent_attr": {"attr": 2}, "override_attr": "2"},
        "parent": {"extends": "grand_parent", "parent_attr": "1", "override_attr": "1"},
    }
    target_json2 = {"extends": "parent", "child_attr": "0", "override_attr": "0"}

    result2 = json_extends(
        whole_json=whole_json2, parent_name="child", target_json=target_json2
    )
    expected2 = {
        "grand_parent_attr": {"attr": 2},
        "parent_attr": "1",
        "child_attr": "0",
        "override_attr": "0",
    }
    assert result2 == expected2

    target_json3 = {"extends": "err_parent", "child_attr": "0", "override_attr": "0"}

    with pytest.raises(ValueError):
        json_extends(
            whole_json=whole_json2, parent_name="child", target_json=target_json3
        )

    whole_json4 = {
        "parent": {"extends": "child", "parent_attr": "1", "override_attr": "1"}
    }
    target_json4 = {"extends": "parent", "child_attr": "0", "override_attr": "0"}

    with pytest.raises(ValueError):
        json_extends(
            whole_json=whole_json4, parent_name="child", target_json=target_json4
        )
