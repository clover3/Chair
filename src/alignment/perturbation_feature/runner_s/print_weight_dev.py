from alignment.perturbation_feature.train_configs import get_pert_train_data_shape
from typing import List, Iterable, Callable, Dict, Tuple, Set

from alignment.perturbation_feature.train_fns import build_model
import numpy as np


def main():
    shape: List[int] = get_pert_train_data_shape()
    model = build_model(shape)
    dense_layer = model.layers[-1]
    print(dense_layer)
    w, b = dense_layer.get_weights()
    print(np.reshape(w, [9, 3]))


if __name__ == "__main__":
    main()