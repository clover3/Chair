from time import strftime, gmtime


def tprint(*arg):
    tim_str = strftime("%H:%M:%S", gmtime())
    all_text = " ".join(str(t) for t in arg)
    print("{} : {}".format(tim_str, all_text))


def main():
    tprint("hi", 1)
    tprint("hi")


if __name__ == "__main__":
    main()
