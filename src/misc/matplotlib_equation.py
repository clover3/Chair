
import matplotlib.pyplot as plt
import numpy as np


def graph(formula, x_range):
    x = np.array(x_range) / 10
    y = formula(x)
    plt.plot(x, y)


def get_formula(k):
    def my_formula(tf):
        return tf * (k + 1) / (tf+k)
    return my_formula

if __name__ == "__main__":

    for k in [0.01, 0.1, 0.75, 1.2]:
        graph(get_formula(k), range(0, 100))
    plt.show()
