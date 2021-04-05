import glob
import os
from collections import Counter
from typing import List, Tuple

from list_lib import lmap, left, right, lflatten

LABEL_NONE = 0
LABEL_AGREE = 1
LABEL_DISAGREE = 2


def parse_file(file_path) -> List[Tuple[str, int]]:
    lines = open(file_path, "r", encoding="utf-8").readlines()
    heads = lines[0].split("\t")
    data = [line.split("\t") for line in lines[1:] if not line.startswith(';')]
    idx_pos = heads.index("positive")
    idx_neg = heads.index("negative")
    idx_text = heads.index("transcript")

    out_data = []
    for line in data:
        text = line[idx_text]
        if line[idx_pos] == 'true':
            label = LABEL_AGREE
        elif line[idx_neg] == 'true':
            label = LABEL_DISAGREE
        else:
            label = LABEL_NONE

        out_data.append((text, label))

    return out_data


def load_all_aawd_alignment() -> List[Tuple[str, int]]:
    files = get_file_itr()
    all_data: List[Tuple[str, int]] = []
    for file_path in files:
        all_data.extend(parse_file(file_path))
    return all_data


def load_aawd_splits_as_binary():
    files = get_file_itr()
    data_list: List[List[Tuple[str, int]]] = lmap(parse_file, files)

    def convert(tuple):
        text, label = tuple
        label = {
            0: 0,
            1: 0,
            2: 1,
        }[label]
        return text, label

    def convertl(tuple_list) -> List[Tuple[str, int]]:
        return lmap(convert, tuple_list)

    dev, test, train = split_train_dev_test(data_list)
    return convertl(train), convertl(dev), convertl(test)


DataSetType = List[Tuple[str, int]]


def load_aawd_splits() -> Tuple[DataSetType, DataSetType, DataSetType]:
    files = get_file_itr()
    data_list: List[List[Tuple[str, int]]] = lmap(parse_file, files)

    dev, test, train = split_train_dev_test(data_list)
    return train, dev, test


def split_train_dev_test(data_list):
    train_len = int(0.8 * len(data_list))
    val_len = int(0.1 * len(data_list))
    test_len = len(data_list) - train_len - val_len
    train = lflatten(data_list[:train_len])
    dev = lflatten(data_list[train_len: train_len + val_len])
    test = lflatten(data_list[train_len + val_len:])
    return dev, test, train


def get_file_itr():
    data_path = "C:\\work\\Data"
    file_path_wildcard = os.path.join(data_path, "AAWD1.1", "1.1", "english", "alignment", "merged", "*.xtdf")
    files = glob.glob(file_path_wildcard)
    return files


def load_train_dev():
    train, dev, test = load_aawd_splits()
    train_x = left(train)
    train_y = right(train)
    dev_x = left(dev)
    dev_y = right(dev)
    return train_x, train_y, dev_x, dev_y


def stat():
    data = load_all_aawd_alignment()
    print(len(data))
    y_labels = right(data)
    print(len(y_labels))
    counter = Counter(y_labels)
    print(counter)


if __name__ == "__main__":
    stat()