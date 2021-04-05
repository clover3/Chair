SEP_ID = 102


def split_p_h_with_input_ids(np_arr, input_ids):
    idx_sep1, idx_sep2 = get_sep_loc(input_ids)
    p = np_arr[1:idx_sep1]
    h = np_arr[idx_sep1 + 1:idx_sep2]
    return p, h


def get_sep_loc(input_ids):
    for i in range(len(input_ids)):
        if input_ids[i] == SEP_ID:
            idx_sep1 = i
            break
    for i in range(idx_sep1 + 1, len(input_ids)):
        if input_ids[i] == SEP_ID:
            idx_sep2 = i
    return idx_sep1, idx_sep2


