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
def explain_by_deletion(data, target_tag, forward_run):
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
    if target_tag == 'conflict':
        logit_attrib_list = logits_list[:, 2] - logits_list[:, 0]
    elif target_tag == 'match':
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


def explain_by_random(data, target_tag, forward_run):
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


class TreeDeletion:
    def __init__(self, paired_info, idx_trans_fn):
        self.info_hash = self.build_hash(paired_info)
        self.idx_trans_fn = idx_trans_fn

    def build_hash(self, paired_info):
        info_hash = dict()
        for entry, info in paired_info:
            code = self.entry_hash(entry)
            info_hash[code] = info
        return info_hash

    def entry_hash(self, entry):
        input_ids = entry[0]
        sig_len = 20
        code = ""
        for c in input_ids[:sig_len]:
            code += str(c)
        return code

    def get_info(self, entry):
        return self.info_hash[self.entry_hash(entry)]

    # score of x[i] is max(drop_j - penalty) for all deletion_j
    # where deletion_j include x[i] as a deleted token
    # penalty = 0.1 * num(deleted_tokens)
    def explain(self, data, target_tag, forward_run):
        runs = []
        runs_info = []
        base_indice = []

        def sample_size():
            prob = [(1, 0.8), (2, 0.2)]
            v = random.random()
            for n, p in prob:
                v -= p
                if v < 0:
                    return n
            return 1

        def penalty(indice):
            return len(indice) * 0.1

        def mask2indice(mask):
            r = []
            for idx in range(len(mask)):
                if mask[idx] == 1:
                    r.append(idx)
            return r

        print("Preparing runs")
        for entry in data:
            info = self.get_info(entry)
            x0, x1, x2 = entry

            # Delete
            base_case = entry
            base_case_idx = len(runs_info)
            base_indice.append(base_case_idx)
            runs.append(base_case)

            seq_len = len(x0)
            real_len = get_real_len(x1, seq_len)
            run_info = {
                'base_case_idx': base_case_idx,
                'type': 'base_run',
                'seq_len' : seq_len,
            }
            runs_info.append(run_info)
            num_trial = real_len
            for _ in range(num_trial):
                x_list, delete_mask = tree_delete(sample_size(), info, self.idx_trans_fn, x0, x1, x2)
                run_info = {
                    'base_case_idx': base_case_idx,
                    'type': 'mod',
                    'mod_mask': delete_mask
                }
                runs.append(x_list)
                runs_info.append(run_info)

        print("Excecuting runs")
        logits_list = forward_run(runs)
        if target_tag == 'conflict':
            logit_attrib_list = logits_list[:, 2] - logits_list[:, 0]
        elif target_tag == 'match':
            logit_attrib_list = logits_list[:, 2] + logits_list[:, 0] - logits_list[:, 1]

        assert len(logits_list) == len(runs)
        print("Summarizing ")
        explains = []
        for case_idx in base_indice:
            base_ce = logit_attrib_list[case_idx]
            idx = case_idx + 1
            seq_len = runs_info[case_idx]['seq_len']
            attrib_scores = np.zeros([seq_len]) - 100
            while idx < len(runs_info) and runs_info[idx]['base_case_idx'] == case_idx:
                after_ce = logit_attrib_list[idx]
                diff_ce = base_ce - after_ce
                delete_mask = runs_info[idx]['mod_mask']
                delete_indice = mask2indice(delete_mask)
                score = diff_ce - penalty(delete_indice)
                for i in delete_indice:
                    attrib_scores[i] = score
                idx += 1

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
