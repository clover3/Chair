from collections import Iterator

from arg.counter_arg.data_loader import load_labeled_data
from arg.counter_arg.header import topics, ArguDatapoint


def main():
    data_itr: Iterator[ArguDatapoint] = load_labeled_data("training", topics[0])
    prev_text1_id = None
    for entry in data_itr:
        if entry.text1.id == prev_text1_id:
            print("text1 is same as before")
            pass
        else:
            print("text1: ", entry.text1)
            prev_text1_id = entry.text1.id
        print("text2: ", entry.text2)
        print(entry.annotations)
        print("------------------")


if __name__ == "__main__":
    main()
