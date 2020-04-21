import os

import numpy as np
from scipy.stats import ttest_ind

from cpath import output_path
from tlm.alt_emb.prediction_analysis import get_correctness


def main():
    file_path = os.path.join(output_path, "nli_tfrecord_cls_300", "dev_mis_alt_small")
    correctness_1 = get_correctness("nli_alt_emb_pred", file_path)
    correctness_2 = get_correctness("alt_emb_G100K", file_path)
    correctness_3 = get_correctness("baseline_alt_emb", file_path)


    print(np.sum(np.equal(correctness_1, correctness_3)))
    print(np.sum(np.equal(correctness_2, correctness_3)))

    print(sum(correctness_1), sum(correctness_2), sum(correctness_3))
    print(ttest_ind(correctness_1, correctness_2))
    print(ttest_ind(correctness_2, correctness_3))



if __name__ == "__main__":
    main()

