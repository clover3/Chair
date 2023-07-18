import sys

from misc_lib import get_first


def main():
    f = open(sys.argv[1], "r")

    missing = []
    for line in f:
        _run, _cand, st, ed = line.split("_")
        if (st, ed) not in missing:
            missing.append((int(st), int(ed)))

    missing.sort(key=get_first)
    for st, ed in missing:
        print((st, ed))



    return NotImplemented


if __name__ == "__main__":
    main()