
import random


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