import sys

from misc.show_checkpoint_vars import load_checkpoint_vars


def show_checkpoint(lm_checkpoint, var_name):
    d = load_checkpoint_vars(lm_checkpoint)
    print(var_name)
    print(d[var_name])


if __name__ == "__main__":
    show_checkpoint(sys.argv[1], sys.argv[2])
