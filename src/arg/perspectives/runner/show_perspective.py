from arg.perspectives.evaluate import perspective_getter


def main():
    while True:
        s = input()
        pid = int(s)
        print(perspective_getter(pid))


if __name__ == "__main__":
    main()
