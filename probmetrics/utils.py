import copy


def join_dicts(*dicts, allow_overlap: bool = True):
    # Attention: arguments do not commute since later dicts can override entries from earlier dicts!
    result = copy.copy(dicts[0])
    for d in dicts[1:]:
        if not allow_overlap and any([key in result for key in d.keys()]):
            raise ValueError(f'Overlapping keys with allow_overlap=False')
        result.update(d)
    return result


