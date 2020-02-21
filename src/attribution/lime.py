import numpy as np
from lime import lime_text

from attribution.baselines import informative_fn_eq1
from data_generator.text_encoder import OOV_ID
from misc_lib import *


def explain_by_lime(data, target_tag, forward_run):
    x0, x1, x2 = data[0]
    len_seq = len(x0)
    print(len_seq)

    def split(s):
        return s.split()

    explainer = lime_text.LimeTextExplainer(split_expression=split, bow=False)
    def signal_f(x):
        logits_list = forward_run(x)
        logit_attrib_list = informative_fn_eq1(target_tag, logits_list)
        return np.reshape(logit_attrib_list, [-1, 1])

    token_map ={}
    token_idx = 3

    def forward_wrap(entry):
        nonlocal token_idx
        x0, x1, x2 = entry
        virtual_tokens = []
        for loc in range(len_seq):
            rt = x0[loc], x1[loc], x2[loc]
            if rt in token_map:
                vt = token_map[rt]
            else:
                token_map[rt] = token_idx
                vt = token_idx
                token_idx = token_idx + 1
            virtual_tokens.append(str(vt))
        return " ".join(virtual_tokens)

    print("Virtualizing data")
    v_data = list([forward_wrap(e) for e in data])
    rev_token_map = dict_reverse(token_map)



    def virtual_forward_run(vtokens_vector):
        def reform(t):
            if t == 'UNKWORDZ':
                return 2
            else:
                return int(t)
        new_inputs = []
        for vstr in vtokens_vector:
            x0 = []
            x1 = []
            x2 = []
            vtokens = [reform(t)for t in vstr.split()]
            for token_idx in vtokens:
                if token_idx == 2:
                    a = OOV_ID
                    b = x1[-1] if x1 else 0
                    c = x2[-1] if x1 else 1
                else:
                    a,b,c = rev_token_map[token_idx]
                x0.append(a)
                x1.append(b)
                x2.append(c)

            new_inputs.append((x0, x1, x2))
        return signal_f(new_inputs)

    explains = []

    print("running lime")
    tick = TimeEstimator(len(v_data))
    for entry in v_data:
        exp = explainer.explain_instance(entry, virtual_forward_run, labels=(0,),
                                         num_features=len_seq, num_samples=500)
        _, scores = zip(*list(exp.local_exp[0]))
        explains.append(scores)
        tick.tick()
    return explains


def explain_by_lime_notag(data, forward_run):

    x0, x1, x2 = data[0]
    len_seq = len(x0)
    def split(s):
        return s.split()

    explainer = lime_text.LimeTextExplainer(split_expression=split, bow=False)
    token_map ={}
    token_idx = 3

    def forward_wrap(entry):
        nonlocal token_idx
        x0, x1, x2 = entry
        virtual_tokens = []
        for loc in range(len_seq):
            rt = x0[loc], x1[loc], x2[loc]
            if rt in token_map:
                vt = token_map[rt]
            else:
                token_map[rt] = token_idx
                vt = token_idx
                token_idx = token_idx + 1
            virtual_tokens.append(str(vt))
        return " ".join(virtual_tokens)


    print("Virtualizing data")
    v_data = list([forward_wrap(e) for e in data])
    rev_token_map = dict_reverse(token_map)



    def virtual_forward_run(vtokens_vector):
        def reform(t):
            if t == 'UNKWORDZ':
                return 2
            else:
                return int(t)
        new_inputs = []
        for vstr in vtokens_vector:
            x0 = []
            x1 = []
            x2 = []
            vtokens = [reform(t)for t in vstr.split()]
            for token_idx in vtokens:
                if token_idx == 2:
                    a = OOV_ID
                    b = x1[-1] if x1 else 0
                    c = x2[-1] if x1 else 1
                else:
                    a,b,c = rev_token_map[token_idx]
                x0.append(a)
                x1.append(b)
                x2.append(c)

            new_inputs.append((x0, x1, x2))
        return forward_run(new_inputs)

    explains = []

    print("running lime")
    tick = TimeEstimator(len(v_data))
    for entry in v_data:
        exp = explainer.explain_instance(entry, virtual_forward_run, num_features=len_seq)
        _, scores = zip(*list(exp.local_exp[0]))
        explains.append(scores)
        tick.tick()
    return explains