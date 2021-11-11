from typing import List, Tuple

import numpy as np

from cache import load_from_pickle
from tlm.qtype.qtype_analysis import QTypeInstance2


def main():
    run_name = "mmd_4U"
    out_entries: List[Tuple[QTypeInstance2, QTypeInstance2]] = load_from_pickle(run_name + "_qtype_parsed")

    e1, e2 = out_entries[0]

    def get_qtype_weights_summary(weights):
        indice = np.argsort(weights)[::-1]
        s_list = []
        for j in indice[:20]:
            s = "{0} ({1:.2f})".format(j, weights[j])
            s_list.append(s)
        line1 = " ".join(s_list)
        for j in indice[-20:]:
            s = "{0} ({1:.2f})".format(j, weights[j])
            s_list.append(s)
        line2 = " ".join(s_list)

        print(line1)
        print(line2)

    get_qtype_weights_summary(e1.qtype_weights_q)


if __name__ == "__main__":
    main()
