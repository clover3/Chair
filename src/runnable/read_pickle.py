import code
import pickle
import sys


def main():
    obj = pickle.load(open(sys.argv[1], "rb"))
    code.interact(local=locals())


if __name__ == "__main__":
    main()