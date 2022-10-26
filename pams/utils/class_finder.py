from typing import Type


def find_class(name: str) -> Type:
    _ = __import__("pams", globals(), locals())
    object_class_candidates = [
        getattr(m, name)
        for m in [*globals().values(), *locals().values()]
        if hasattr(m, name)
    ]
    if len(object_class_candidates) != 1:
        raise AttributeError(
            f"class for {name} is found {len(object_class_candidates)} times"
        )
    object_class: Type = object_class_candidates[0]
    return object_class
