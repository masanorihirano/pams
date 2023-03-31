from typing import Dict
from typing import List
from typing import Optional


def json_extends(
    whole_json: Dict,
    parent_name: str,
    target_json: Dict,
    excludes_fields: Optional[List[str]],
) -> Dict:
    """extend target json. If "extends" keys are included in target_json, this try to extend the json dict recursively.
     For the extension, the value for "extends" field  is found in whole_json (only under parent_name is targeted) and
     target_json is extended using the dict under the found part of whole_json. This process is recursively repeated.

    Args:
        whole_json (Dict): whole of json.
        parent_name (str): parent name of json.
        target_json (Dict): target json.
        excludes_fields (List[str], Optional): exclude list from json field.

    Returns:
        Dict: json output.
    """
    # ToDo: add a code example to doc.
    excludes_fields_: List[str] = excludes_fields if excludes_fields is not None else []
    results = target_json.copy()
    extending_history: List[str] = [parent_name]
    while "extends" in results:
        extends_class: str = results["extends"]
        results.pop("extends")
        if extends_class not in whole_json:
            raise ValueError(
                f"{parent_name} cannot extends {extends_class} because {extends_class} is missing"
            )
        if extends_class in extending_history:
            raise ValueError(f"{parent_name} has extending loop")
        extending_history.append(extends_class)
        extending_dict: Dict = whole_json[extends_class]
        results = dict(
            [
                (key, value)
                for key, value in extending_dict.items()
                if key not in excludes_fields_
            ],
            **results,
        )
    return results
