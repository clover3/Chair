
import random
from data_generator.NLI.parse_tree import *
from data_generator.NLI import nli

def token_delete(binary_tag, x0, x1, x2):
    assert len(x0) == len(x1)
    assert len(x0) == len(x2)
    assert len(x0) == len(binary_tag)

    length = len(binary_tag)
    x0_new = []
    x1_new = []
    x2_new = []

    for i in range(length):
        if not binary_tag[i]:
            x0_new.append(x0[i])
            x1_new.append(x1[i])
            x2_new.append(x2[i])

    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)
    return x0_new, x1_new, x2_new


def token_delete_with_indice(indice, x0, x1, x2):
    mask = [0] * len(x0)
    for idx in indice:
        mask[idx] = 1

    return token_delete(mask, x0, x1, x2)


def token_replace(binary_tag, x0, x1, x2, random_token):
    assert len(x0) == len(x1)
    assert len(x0) == len(x2)
    assert len(x0) == len(binary_tag)

    length = len(binary_tag)
    x0_new = []
    x1_new = []
    x2_new = []

    for i in range(length):
        if not binary_tag[i]:
            x0_new.append(x0[i])
            x1_new.append(x1[i])
            x2_new.append(x2[i])
        else:
            x0_new.append(random_token())
            x1_new.append(x1[i])
            x2_new.append(x2[i])

    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)
    return x0_new, x1_new, x2_new



def random_delete(num_del, x0, x1, x2):
    num_del = max(num_del, 1)
    length = len(x0)

    last_valid = 0
    for i in range(length):
        if x2[i] > 0 :
            last_valid = i
    num_del = min(num_del, last_valid)

    del_indice = random.sample(range(last_valid+1), num_del)
    x0_new = []
    x1_new = []
    x2_new = []
    mask = []

    for i in range(length):
        if i not in del_indice:
            x0_new.append(x0[i])
            x1_new.append(x1[i])
            x2_new.append(x2[i])
            mask.append(0)
        else:
            mask.append(1)

    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)
    x_list = x0_new, x1_new, x2_new
    return x_list, mask


def seq_replace_inner(targ_mask, src_loc, x0, x1, x2):
    length = len(x0)
    x0_new = []
    x1_new = []
    x2_new = []

    f_first_del = True

    for i in range(length):
        if len(x0_new) >= length:
            break

        if not targ_mask[i]:
            x0_new.append(x0[i])
            x1_new.append(x1[i])
            x2_new.append(x2[i])
        else:
            if f_first_del:
                for idx in src_loc:
                    x0_new.append(x0[idx])
                    x1_new.append(x1[i])
                    x2_new.append(x2[i])
                    if len(x0_new) >= length:
                        break
                f_first_del = False

    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)

    assert len(x0_new) == length
    assert len(x1_new) == length
    assert len(x2_new) == length
    return x0_new, x1_new, x2_new



def seq_add(src_indice, targ_loc, x0, x1, x2):
    length = len(x0)
    x0_new = []
    x1_new = []
    x2_new = []

    for i in range(length):
        if len(x0_new) >= length:
            break

        if i == targ_loc:
            for idx in src_indice:
                x0_new.append(x0[idx])
                x1_new.append(x1[i])
                x2_new.append(x2[i])
                if len(x0_new) >= length:
                    break

        x0_new.append(x0[i])
        x1_new.append(x1[i])
        x2_new.append(x2[i])


    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)

    assert len(x0_new) == length
    assert len(x1_new) == length
    assert len(x2_new) == length
    return x0_new, x1_new, x2_new




def seq_replace(num_del, mark_loc, x0, x1, x2 ):
    length = len(x0)

    for i in range(length):
        if x0[i] == nli.SEP_ID:
            idx_sep1 = i
            break

    is_prem_mark = mark_loc[0] < idx_sep1

    last_valid = 0
    for i in range(length):
        if x2[i] > 0 :
            last_valid = i
    num_del = min(num_del, last_valid)
    if is_prem_mark:
        idx_sel_start = idx_sep1 + 1
        idx_sel_end = last_valid + 1
    else:
        idx_sel_start = 1
        idx_sel_end = idx_sep1


    def sample_len():
        l = 1
        v = random.random()
        while v < 0.3 and l < length:
            l = l * 2
        return min(l, length)

    def sample_start():
        return pick1(range(idx_sel_start, idx_sel_end))


    indice = []
    for i in range(num_del):
        del_len = sample_len()
        start_idx = sample_start()
        end_idx = min(start_idx + del_len, idx_sel_end)
        for idx in range(start_idx, end_idx):
            indice.append(idx)

    mask = [0] * len(x0)
    for idx in indice:
        mask[idx] = 1

    return token_delete(mask, x0, x1, x2), seq_replace_inner(mask, mark_loc, x0, x1, x2), mask


def add_seq_hypo2prem(x0, x1, x2 ):
    length = len(x0)

    for i in range(length):
        if x0[i] == nli.SEP_ID:
            idx_sep1 = i
            break


    last_valid = 0
    for i in range(length):
        if x2[i] > 0 :
            last_valid = i
    idx_sel_start = idx_sep1 + 1
    idx_sel_end = last_valid


    def sample_len():
        l = 1
        v = random.random()
        while v < 0.8 and l < length:
            l = l * 2
        return min(l, length)

    def sample_start():
        return pick1(range(idx_sel_start, idx_sel_end))


    mask = [0] * len(x0)
    src_indice = []
    del_len = sample_len()
    start_idx = sample_start()
    end_idx = min(start_idx + del_len, idx_sel_end)
    for idx in range(start_idx, end_idx):
        src_indice.append(idx)
        mask[idx] = 1

    targ_loc = pick1(range(1, idx_sep1+1))

    return seq_add(src_indice, targ_loc, x0, x1, x2), mask


def seq_delete(num_del, info, idx_trans_fn, x0, x1, x2):
    length = len(x0)
    last_valid = 0
    for i in range(length):
        if x2[i] > 0 :
            last_valid = i
    num_del = min(num_del, last_valid)

    def sample_len():
        l = 1
        v = random.random()
        while v < 0.5 and l < length:
            l = l * 2
            v = random.random()
        return min(l, length)

    indice = []
    for i in range(num_del):
        del_len = sample_len()
        start_idx = pick1(range(last_valid+1))
        end_idx = min(start_idx+del_len, last_valid+1)
        for idx in range(start_idx, end_idx):
            indice.append(idx)

    mask = [0] * len(x0)
    for idx in indice:
        mask[idx] = 1

    return token_delete(mask, x0, x1, x2), mask

def tree_delete(num_del, info, idx_trans_fn, x0, x1, x2):
    s1, s2, bp1, bp2 = info

    tree_prem = binary_parse_to_tree(bp1)
    tree_hypo = binary_parse_to_tree(bp2)
    input_x = x0, x1, x2

    def sample_any():
        sel = random.randint(0,1)
        tree = [tree_prem, tree_hypo][sel]

        node = tree.sample_any_node()
        indice = node.all_leave_idx()
        parse_tokens = tree.leaf_tokens()
        return idx_trans_fn(parse_tokens, input_x, indice, sel)

    def sample_leave():
        idx = random.randint(0, 1)
        tree = [tree_prem, tree_hypo][idx]
        node = tree.sample_leaf_node()
        return node.all_leave_idx()


    indice = []
    for i in range(num_del):
        indice += sample_any()

    mask = [0] * len(x0)
    for idx in indice:
        mask[idx] = 1

    return token_delete(mask, x0, x1, x2), mask


def random_delete_with_mask(num_del, x0, x1, x2, q_mask):
    num_del = max(num_del, 1)
    length = len(x0)

    sample_space = []
    for i in range(length):
        if q_mask[i] > 0 :
            sample_space.append(i)
    num_del = min(num_del, len(sample_space))

    del_indice = random.sample(sample_space, num_del)
    x0_new = []
    x1_new = []
    x2_new = []
    mask = []

    for i in range(length):
        if i not in del_indice:
            x0_new.append(x0[i])
            x1_new.append(x1[i])
            x2_new.append(x2[i])
            mask.append(0)
        else:
            mask.append(1)

    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)
    x_list = x0_new, x1_new, x2_new
    return x_list, mask




def random_replace_with_mask(num_del, x0, x1, x2, q_mask, random_token):
    num_del = max(num_del, 1)
    length = len(x0)

    sample_space = []
    for i in range(length):
        if q_mask[i] > 0 :
            sample_space.append(i)
    num_del = min(num_del, len(sample_space))

    del_indice = random.sample(sample_space, num_del)
    x0_new = []
    x1_new = []
    x2_new = []
    mask = []

    for i in range(length):
        if i not in del_indice:
            x0_new.append(x0[i])
            x1_new.append(x1[i])
            x2_new.append(x2[i])
            mask.append(0)
        else:
            x0_new.append(random_token())
            x1_new.append(x1[i])
            x2_new.append(x2[i])
            mask.append(1)

    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)
    x_list = x0_new, x1_new, x2_new
    return x_list, mask