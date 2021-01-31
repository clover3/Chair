import sys


def main():

    def read_lines(path):
        s = set()
        for line in open(path, "r"):
            s.add(line.strip())
        return s

    l1 = set(read_lines(sys.argv[1]))
    l2 = set(read_lines(sys.argv[2]))

    print("1: ", len(l1))
    print("2: ", len(l2))
    common = l1.intersection(l2)
    print("common:", len(common))


if __name__ == "__main__":
    main()