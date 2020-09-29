from arg.counter_arg.eval import load_problems
from arg.counter_arg.header import splits


def main():
    d = {}
    for split in splits:
        d[split] = len(load_problems(split))

    print(d)


if __name__ == "__main__":
    main()