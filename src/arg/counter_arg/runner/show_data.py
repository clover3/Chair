from collections import Iterator, defaultdict

from arg.counter_arg.data_loader import load_labeled_data_per_topic
from arg.counter_arg.header import topics, ArguDataPoint


def main():
    data_itr: Iterator[ArguDataPoint] = load_labeled_data_per_topic("training", topics[0])
    prev_text1_id = None

    group_by_text_1 = defaultdict(list)
    text1_dict = {}

    for entry in data_itr:
        if entry.text1.id not in group_by_text_1:
            group_by_text_1[entry.text1.id] = []

        label_A = entry.annotations[0] == 'true'
        label_B = entry.annotations[2] == 'true'
        group_by_text_1[entry.text1.id].append((entry.text2, label_A, label_B))
        text1_dict[entry.text1.id] = entry.text1.text

    for text_id in group_by_text_1:
        def print_items(condition):
            for item in group_by_text_1[text_id]:
                text, label_A, label_B = item
                if condition(label_A, label_B):
                    print(text)

        print("> Text1 : ")
        print(text1_dict[text_id])
        print("> True , True cases")
        print_items(lambda a, b: a and b)
        print("> False, True cases")
        print_items(lambda a, b: not a and b)
        print("> True, False cases")
        print_items(lambda a, b: a and not b)
        print("> False cases")
        print_items(lambda a, b: not a and not b)


if __name__ == "__main__":
    main()
