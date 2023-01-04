from typing import List, Iterable, Callable, Dict, Tuple, Set, NamedTuple

from cache import load_list_from_jsonl
from cpath import at_data_dir
from cpath import output_path
from iter_util import load_jsonl
from misc_lib import path_join


def get_corpus_dir():
    return at_data_dir("SciTailV1.1", "predictor_format")


def get_scitail_jsonl_path(split):
    return path_join(get_corpus_dir(), f"scitail_1.0_structure_{split}.jsonl")

scitail_label_list = ["entails", "neutral"]


class ScitailEntry(NamedTuple):
    question: str
    sentence1: str
    sentence2: str
    label: str

    def get_label_as_int(self):
        return scitail_label_list.index(self.label)

    @classmethod
    def from_json(cls, j):
        return ScitailEntry(
            j['question'],
            j['sentence1'],
            j['sentence2'],
            j['gold_label']
        )


def load_scitail_jsonl(split) -> List[ScitailEntry]:
    return load_list_from_jsonl(get_scitail_jsonl_path(split), ScitailEntry.from_json)


