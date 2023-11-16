import pickle


def main():
    obj = pickle.load(open("/tmp/pickle", "rb"))
    print(obj)


if __name__ == "__main__":
    main()