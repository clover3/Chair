import os
import sys

import numpy as np

from cpath import output_path
from misc.show_checkpoint_vars import load_checkpoint_vars


def main():
    start = os.path.join(output_path, "model", "runs", "nli_ex_21", "model-73630")
    modified = os.path.join(output_path, "model", "runs", "nli_pairing_1", "model-11171")
    start = sys.argv[1]
    modified = sys.argv[2]

    var_d1 = load_checkpoint_vars(start)
    var_d2 = load_checkpoint_vars(modified)

    for key in var_d1:
        if key in var_d2:
            v1 = var_d1[key]
            v2 = var_d2[key]

            print(key, np.sum(v1 - v2))


if __name__ == "__main__":
    main()