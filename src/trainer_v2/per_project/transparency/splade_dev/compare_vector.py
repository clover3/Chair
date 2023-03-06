import pickle
import numpy as np
from cache import load_from_pickle


def main():
    tf_out = load_from_pickle("tf_splade_out")
    torch_out = load_from_pickle("torch_splade_out")

    error = np.sum(tf_out - torch_out)
    print(error)



if __name__ == "__main__":
    main()