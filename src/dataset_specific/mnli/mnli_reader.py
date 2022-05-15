import os
from typing import NamedTuple, Iterator

from data_generator.NLI.enlidef import nli_label_list
from data_generator.NLI.nli_info import corpus_dir


class NLIPairData(NamedTuple):
    premise: str
    hypothesis: str
    label: str
    data_id: str

    def get_label_as_int(self):
        return nli_label_list.index(self.label)


def parse_nli_line(line) -> NLIPairData:
    split_line = line.split("\t")
    s1, s2 = split_line[8:10]
    return NLIPairData(
        premise=s1,
        hypothesis=s2,
        label=split_line[-1],
        data_id=split_line[0]
    )


def iter_lines_skip_first(file_path):
    for idx, line in enumerate(open(file_path, "rb")):
        if idx == 0: continue  # skip header
        line = line.strip().decode("utf-8")
        yield line


class MNLIReader:
    def __init__(self):
        self.train_file = os.path.join(corpus_dir, "train.tsv")
        self.dev_file = os.path.join(corpus_dir, "dev_matched.tsv")
        self.split_file_path = {
            'train': self.train_file,
            'dev': self.dev_file
        }

    def get_train(self) -> Iterator[NLIPairData]:
        return self.load_split("train")

    def get_dev(self) -> Iterator[NLIPairData]:
        return self.load_split("dev")

    def load_split(self, split_name) -> Iterator[NLIPairData]:
        line_itr = iter_lines_skip_first(self.split_file_path[split_name])
        for line in line_itr:
            yield parse_nli_line(line)

