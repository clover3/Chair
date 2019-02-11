
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