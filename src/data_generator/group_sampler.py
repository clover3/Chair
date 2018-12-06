import random


class KeySampler:
    def __init__(self, group_dict):
        self.g2 = []
        self.g = list(group_dict.keys())
        for key, items in group_dict.items():
            if len(items) > 3:
                self.g2.append(key)

    # Sample from all group
    def sample(self):
        end = len(self.g) - 1
        return self.g[random.randint(0, end)]

    # Sample from groups except first size
    def sample2(self):
        end = len(self.g2) - 1
        return self.g2[random.randint(0, end)]

# grouped_dict : dict(key->list)

def pos_neg_pair_sampling(grouped_dict, key_sampler, target_size):
    # from default setting, only group with more than 3 items will be sampled


    def sample_key(except_key = None):
        sampled_key = key_sampler.sample()
        while except_key is not None and except_key == sampled_key:
            sampled_key = key_sampler.sample()
        return sampled_key

    LABEL_POS = 1
    LABEL_NEG = 0
    pos_size = int(target_size / 2)
    neg_size = target_size - pos_size

    data = []
    count = 0
    while count < pos_size:
        key = key_sampler.sample2()
        items = grouped_dict[key]
        i1, i2 = random.sample(range(len(items)), 2)
        data.append((items[i1], items[i2], LABEL_POS))
        count += 1

    for i in range(neg_size):
        key = key_sampler.sample()
        items = grouped_dict[key]
        item1 = items[random.randint(0, len(items)-1)]

        item_2_group = sample_key(key)
        l2 = grouped_dict[item_2_group]
        item_2_idx = random.randint(0, len(l2)-1)
        item2 = l2[item_2_idx]
        data.append((item1, item2, LABEL_NEG))

    assert len(data) == target_size
    random.shuffle(data)
    return data


def pos_sampling(grouped_dict, key_sampler, target_size):
    data = []
    count = 0
    dummy = 0
    while count < target_size:
        key = key_sampler.sample2()
        items = grouped_dict[key]
        i1, i2 = random.sample(range(len(items)), 2)
        data.append((items[i1], items[i2]))
        count += 1

    return data