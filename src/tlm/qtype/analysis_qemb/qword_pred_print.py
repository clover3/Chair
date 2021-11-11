from collections import Counter
from typing import List

import numpy as np
from tlm.qtype.analysis.save_parsed import parse_q_weight_output

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.qtype_analysis import QTypeInstance


def visualize_dim_analysis(all_insts: List[QTypeInstance]):
    n_cluster = 30522
    clusters = [list() for _ in range(n_cluster)]
    for j, inst in enumerate(all_insts):
        qtype_weights = inst.qtype_weights
        ranked_dims = np.argsort(qtype_weights)[::-1]
        for d in ranked_dims[:10]:
            if qtype_weights[d] > 0.2:
                c: List = clusters[d]
                c.append(j)

    tokenizer = get_tokenizer()
    voca = tokenizer.inv_vocab
    for c_idx, c in enumerate(clusters):
        if not len(c):
            continue

        print(">> Cluster {} : {}".format(c_idx, voca[c_idx]))
        s_counter = Counter()
        for j in c:
            s = all_insts[j].summary()
            s_counter[s] += 1
        for s, cnt in s_counter.most_common():
            print(cnt, s)



def main():
    raw_prediction_path = at_output_dir("qtype", "mmd_qtype_K")
    all_insts = parse_q_weight_output(raw_prediction_path)
    visualize_dim_analysis(all_insts)


if __name__ == "__main__":
    main()
