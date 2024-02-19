import pickle
import random
from collections import defaultdict, Counter
import sys
from adhoc.resource.scorer_loader import get_rerank_scorer
from cpath import output_path
from misc_lib import path_join, average

import nltk
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
from dataset_specific.msmarco.passage.path_helper import get_train_triples_small_path
from misc_lib import pick1
from tab_print import tab_print_dict
from table_lib import tsv_iter
from nltk.tokenize import word_tokenize


def iter_cand_corpus(n_max=100) -> Iterator[Tuple[list[str], list[str], int]]:
    qd_iter = tsv_iter(get_train_triples_small_path())
    random.seed(0)
    cnt = 0
    for q, dp, _dn in qd_iter:
        q_tokens = word_tokenize(q)
        d_tokens = word_tokenize(dp)
        tagged = nltk.pos_tag(d_tokens)

        q_tokens_l = [t.lower() for t in q_tokens]
        qf = Counter(q_tokens_l)
        cand_idx = []
        for idx, t in enumerate(d_tokens):
            if qf[t] == 1:
                word, pos = tagged[idx]
                if pos == "NN" or pos == "NNS":
                    cand_idx.append(idx)

        if cand_idx:
            idx = pick1(cand_idx)
            yield q_tokens, d_tokens, idx
            cnt += 1
            if cnt > n_max:
                break


def main():
    def enum_chars():
        st = ord('a')
        ed = ord('z')
        for i in range(st, ed+1):
            yield chr(i)

    # iterate relevance documents
    #    Find exact match, which is noun
    #    Append a ~ z to the term
    #    Measure score change

    method_name = sys.argv[1]
    reranker = get_rerank_scorer(method_name)
    score_fn = reranker.score_fn
    save_path = path_join(output_path, "mmp", "append_test", method_name)
    observe_d = defaultdict(list)
    for q_tokens, d_tokens, sel_idx in iter_cand_corpus(500):
        payload = [(" ".join(q_tokens), " ".join(d_tokens), "orig")]

        for ch in enum_chars():
            new_d = []
            for idx, t in enumerate(d_tokens):
                if idx == sel_idx:
                    new_t = t + ch
                    new_d.append(new_t)
                else:
                    new_d.append(t)
            payload.append((" ".join(q_tokens), " ".join(new_d), ch))

        payload_qd = [(x[0], x[1]) for x in payload]
        payload_ch = [x[2] for x in payload]
        scores = score_fn(payload_qd)
        orig_scores = scores[0]
        # print(payload_qd)

        for s, role in zip(scores, payload_ch):
            diff = orig_scores - s
            observe_d[role].append(diff)

        # break

    summary_d = {k: average(l) for k, l in observe_d.items()}
    tab_print_dict(summary_d)

    pickle.dump(observe_d, open(save_path, "wb"))


if __name__ == "__main__":
    main()
