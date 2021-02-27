import json
import pickle
import sys


def main():
    obj = json.load(open(sys.argv[1], "r"))
    pickle.dump(obj, open(sys.argv[2], "wb"))


if __name__ == "__main__":
    main()
