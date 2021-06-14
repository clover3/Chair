import sys

from cache import load_pickle_from


def main():
    data = load_pickle_from(sys.argv[1])

    for batch in data:
        logits = batch["logits"]
        print(logits.shape)




    return NotImplemented


if __name__ == "__main__":
    main()