import sys


def main():
    f = open(sys.argv[1], "r")
    loc = int(sys.argv[2])
    f.seek(loc - 1)
    print(f.read(100))


if __name__ == "__main__":
    main()