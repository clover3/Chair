import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def normalize(x):
    """Compute softmax values for each sets of scores in x."""
    return x / x.sum(axis=0) # only difference


def main():
    scores = np.array([0.1, 0.1, 0.9, 0.1])

    def print_scores(scores):
        print("score", scores)
        print("softmax", softmax(scores))
        print("softmax: sum", np.sum(softmax(scores) * scores))
        print("normalize", normalize(scores))
        print("normalize: sum", np.sum(normalize(scores) * scores))

    print_scores(scores)
    print_scores(np.array([0.1, 0.1, 0.6, 0.6]))


if __name__ == "__main__":
    main()