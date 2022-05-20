import numpy as np

from trainer_v2.keras_fit.assymetric_model import fuzzy_logic


def main():
    n_batch = 4
    n_seg = 8
    n_label = 3
    logits = np.zeros([n_batch, n_seg, n_label])
    output = fuzzy_logic(logits)
    print(output)


if __name__ == "__main__":
    main()