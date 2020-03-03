def flatten_nested_list(l) -> iter:
    for el in l:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            for sub in flatten_nested_list(el):
                yield sub
        else:
            yield el
