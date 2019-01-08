
import numpy as np
from .deleter_trsfmr import *
from trainer.tf_module import *
from collections import Counter

def get_real_len(x1, seq_len):
    for i in range(seq_len):
        if x1[i] == 0:
            return i
    return seq_len

# forward_run :
#  input : list of [x0,x1,x2]
#  output: list of softmax val
def explain_by_deletion(data, target_label, forward_run):
    inputs = []
    inputs_info = []
    base_indice = []
    for entry in data:
        x0, x1, x2 = entry
        base_case = entry
        base_case_idx = len(inputs_info)
        base_indice.append(base_case_idx)
        inputs.append(base_case)

        seq_len = len(x0)
        real_len = get_real_len(x1, seq_len)
        info = {
            'base_case_idx': base_case_idx,
            'type': 'base_run',
            'seq_len' : seq_len,
        }
        inputs_info.append(info)

        for idx in range(real_len):
            mask = np.zeros([seq_len])
            mask[idx] = 1
            new_case = token_delete(mask, x0,x1,x2)
            info = {
                'base_case_idx': base_case_idx,
                'type': 'mod',
                'mod_idx': idx
            }
            inputs.append(new_case)
            inputs_info.append(info)

    logits_list = forward_run(inputs)
    if target_label == 'conflict':
        logit_attrib_list = logits_list[:, 2] - logits_list[:, 0]
    elif target_label == 'match':
        logit_attrib_list = logits_list[:, 2] + logits_list[:, 0] - logits_list[:, 1]

    assert len(logits_list) == len(inputs)

    explains = []
    for case_idx in base_indice:
        base_ce = logit_attrib_list[case_idx]
        idx = case_idx + 1
        seq_len = inputs_info[case_idx]['seq_len']
        attrib_scores = np.zeros([seq_len]) - 100
        while idx < len(inputs_info) and inputs_info[idx]['base_case_idx'] == case_idx:
            after_ce = logit_attrib_list[idx]
            diff_ce = base_ce - after_ce
            score = diff_ce
            mod_idx = inputs_info[idx]['mod_idx']
            attrib_scores[mod_idx] = score
            idx += 1

        explains.append(attrib_scores)
    return explains


def explain_by_random(data, target_label, forward_run):
    explains = []

    for entry in data:
        x0, x1, x2 = entry
        seq_len = len(x0)
        real_len = get_real_len(x1, seq_len)
        attrib_scores = np.zeros([seq_len])
        for i in range(real_len):
            attrib_scores[i] = random.random()
        explains.append(attrib_scores)
    return explains


class IdfScorer:
    def __init__(self, batches):
        self.df = Counter()
        for batch in batches:
            x0, x1, x2, y = batch
            batch_size, seq_len = x0.shape
            for b in range(batch_size):
                term_count = Counter()
                for i in range(seq_len):
                    term_count[x0[b,i]] = 1
                for elem, cnt in term_count.items():
                    self.df[elem] += 1


    def explain(self, data, target_label, forward_run):
        explains = []
        for entry in data:
            x0, x1, x2 = entry
            seq_len = len(x0)
            real_len = get_real_len(x1, seq_len)
            attrib_scores = np.zeros([seq_len])
            for i in range(real_len):
                df = self.df[x0[i]]
                df = df if df > 0 else 1
                attrib_scores[i] = 1 / df
            explains.append(attrib_scores)
        return explains
