import os

import numpy as np
from scipy.stats import ttest_ind

from cpath import output_path
from tlm.alt_emb.prediction_analysis import get_correctness


def main():
    file_path = os.path.join(output_path, "nli_tfrecord_cls_300", "dev_mis_alt_small")

    run_names = ["nli_alt_emb_pred",
                 "alt_emb_G100K",
                 "baseline_alt_emb",
                 "alt_emb_H20K",
                 "baseline_clueweb_small",
                 "alt_emb_H100K",
                 ]


    correctness_list = list([get_correctness(name, file_path) for name in run_names])
    typical_len = len(correctness_list[3])
    correctness_list[4] = correctness_list[4][:typical_len]

    print("name  acc  num_correct num_total")
    for name, correctness in zip(run_names, correctness_list):
        c = correctness
        print(name, np.average(c), sum(c), len(c))

    def pair_ttest(idx1, idx2):
        print("{} vs {}".format(run_names[idx1], run_names[idx2]))
        print(ttest_ind(correctness_list[idx1], correctness_list[idx2]))

    pair_ttest(0, 1)
    pair_ttest(1, 2)
    pair_ttest(3, 4)
    pair_ttest(2, 5)


if __name__ == "__main__":
    main()

