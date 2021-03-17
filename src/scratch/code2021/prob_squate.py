import numpy as np


def main():
    a = np.array([0.8, 0.1, 0.1])
    for n in [0.3, 1, 2, 4, 8]:
        t = np.exp(a*n)
        normal = np.sum(t)
        print(t / normal)


if __name__ == "__main__":
    main()