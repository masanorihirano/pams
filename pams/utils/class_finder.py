from typing import List
from typing import Optional
from typing import Type
from typing import Union


def find_class(
    name: Union[str, Type], optional_class_list: Optional[List[Type]] = None
) -> Type:
    """Find class from pams name spaces.

    If a class is given instead of a class name, the class is returned as is without searching.

    .. seealso::
        If you want to use user-defined classes, please use :meth:`pams.runners.Runner.class_register`
        or set the class itself instead of its name.

    Args:
        name (Union[str, Type]): class name or class.
        optional_class_list (List[Type], Optional): optional class list.

    Returns:
        Type: class type.

    Examples:
        >>> from pams.agents import FCNAgent
        >>> from pams.utils import find_class
        >>> find_class(name="FCNAgent") is FCNAgent
        True
        >>> find_class(name=FCNAgent) is FCNAgent
        True
    """
    if isinstance(name, type):
        return name
    if not isinstance(name, str):
        raise ValueError(
            f"class must be a class name (str) or a class, but {name!r} is given"
        )
    _1 = __import__("pams", globals(), locals())
    _2 = __import__("pams.agents", globals(), locals(), ["*"])
    _3 = __import__("pams.events", globals(), locals(), ["*"])
    _4 = __import__("pams.logs", globals(), locals(), ["*"])
    _5 = __import__("pams.utils", globals(), locals(), ["*"])

    candidates_spaces = [*globals().values(), *locals().values()]

    object_class_candidates = [
        getattr(m, name) for m in candidates_spaces if hasattr(m, name)
    ]
    if optional_class_list is not None:
        object_class_candidates.extend(
            [x for x in optional_class_list if x.__name__ == name]
        )
    if len(object_class_candidates) != 1:
        raise AttributeError(
            f"class for {name} is found {len(object_class_candidates)} times"
        )
    object_class: Type = object_class_candidates[0]
    return object_class
