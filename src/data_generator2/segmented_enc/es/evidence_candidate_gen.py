import random
from typing import List, Callable, Iterable, Dict, Tuple, NamedTuple
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es.common import PHSegmentedPair
from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location
from dataset_specific.mnli.mnli_reader import NLIPairData
from misc_lib import ceil_divide


def pool_delete_indices(num_del, seq_len, g) -> List[int]:
    num_del = min(num_del, seq_len)

    def sample_len():
        l = 1
        v = random.random()
        while v < g and l < seq_len:
            l = l * 2
            v = random.random()
        return min(l, seq_len)

    indice = []
    for i in range(num_del):
        del_len = sample_len()
        start_idx = random.randint(0, seq_len-1)
        end_idx = min(start_idx+del_len, seq_len)
        for idx in range(start_idx, end_idx):
            indice.append(idx)
    return indice


class EvidenceCandidateGenerator:
    def __init__(self, num_candidate):
        self.tokenizer = get_tokenizer()
        self.k = num_candidate

    def generate(self, nli_data_itr):
        tokenizer = self.tokenizer
        k = self.k
        for item in nli_data_itr:
            # Split h into two  -> h1, h2
            #       (p, h1), (p, h2)
            #       (p1^(1), h1), (p2^(1), h2),
            #       .....
            #       (p1^(k), h1), (p2^(k), h2),
            p_tokens = tokenizer.tokenize(item.premise)
            h_tokens = tokenizer.tokenize(item.hypothesis)
            h_st, h_ed = get_random_split_location(h_tokens)

            base = PHSegmentedPair(p_tokens, h_tokens, h_st, h_ed, [], [], item)
            yield base
            for _ in range(k):
                p_del_indices1 = self.delete_some(p_tokens)
                p_del_indices2 = self.delete_some(p_tokens)
                yield PHSegmentedPair(p_tokens, h_tokens, h_st, h_ed, p_del_indices1, p_del_indices2, item)

    def delete_some(self, tokens) -> List[int]:
        g = 0.5
        g_inv = int(1 / g)
        max_del = ceil_divide(len(tokens), g_inv)
        num_del = random.randint(1, max_del)
        return pool_delete_indices(num_del, len(tokens), g)

