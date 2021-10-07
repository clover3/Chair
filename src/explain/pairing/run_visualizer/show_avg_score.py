from collections import defaultdict

import numpy as np

from cache import load_pickle_from
from cpath import at_output_dir
from misc_lib import DictValueAverage
from trainer.np_modules import sigmoid


# average options
# 1. any tokens
# 2. Non padding tokens
# 3. Neutral and non-padding


def main():
    num_layers = 12
    dva = DictValueAverage()

    all_val = defaultdict(list)
    for i in range(1):
        save_path = at_output_dir("lms_scores", str(i) + ".pickle")
        output_d = load_pickle_from(save_path)
        input_mask = output_d['input_mask'] # [num_inst, seq_length]
        for layer_no in range(num_layers):
            probs = sigmoid(output_d['logits'][layer_no]) # [num_inst, seq_length, 2]
            num_inst, seq_length, maybe_2 = np.shape(probs)

            for data_idx in range(num_inst):
                for seq_idx in range(seq_length):
                    if input_mask[data_idx, seq_idx]:
                        key = layer_no
                        v = probs[data_idx, seq_idx, 1]
                        dva.add(key, v)
                        all_val[key].append(v)

    for k, v in dva.all_average().items():
        print(k, v)

    for k, l in all_val.items():
        min_val = max(l)
        print(k, min_val)


if __name__ == "__main__":
    main()