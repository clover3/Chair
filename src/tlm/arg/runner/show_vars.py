import sys

from misc.show_checkpoint_vars import load_checkpoint_vars


def main():
    d = load_checkpoint_vars(sys.argv[1])
    var_list = [
        "dense_75/kernel",
        "dense_75/bias",
        "dense_77/kernel",
        "dense_77/bias",
        "k1",
        "k2",
        "bias"
    ]
    for var in var_list:
        print(var, d[var])

if __name__ == "__main__":
    main()