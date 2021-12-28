SEP_ID = 102


class SEPNotFound(IndexError):
    pass


def split_p_h_with_input_ids(np_arr, input_ids):
    idx_sep1, idx_sep2 = get_sep_loc(input_ids)
    p = np_arr[1:idx_sep1]
    h = np_arr[idx_sep1 + 1:idx_sep2]
    return p, h


def get_sep_loc(input_ids):
    idx_sep1 = None
    for i in range(len(input_ids)):
        if input_ids[i] == SEP_ID:
            idx_sep1 = i
            break
    if idx_sep1 is None:
        raise SEPNotFound()

    idx_sep2 = None
    for i in range(idx_sep1 + 1, len(input_ids)):
        if input_ids[i] == SEP_ID:
            idx_sep2 = i
    if idx_sep2 is None:
        raise SEPNotFound()

    return idx_sep1, idx_sep2


def get_first_seg(input_ids):
    return get_first_seg_ex(input_ids, input_ids)


def get_first_seg_ex(np_arr, input_ids):
    for i in range(len(input_ids)):
        if input_ids[i] == SEP_ID:
            idx_sep1 = i
            break
    p = np_arr[1:idx_sep1]
    return p