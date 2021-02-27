import json
import sys


def main():
    json.load(open(sys.argv[1], "r", encoding="utf-8"))


if __name__ == "__main__":
    main()
