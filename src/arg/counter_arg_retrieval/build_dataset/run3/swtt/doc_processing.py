import os
from typing import List, Tuple

from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.window_enum_policy import WindowEnumPolicy
from cache import save_to_pickle, load_from_pickle
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from galagos.swtt_processor import jsonl_to_swtt
from misc_lib import Averager


def main():
    file_path = os.path.join(output_path, "ca_building", "run3", "docs.jsonl")
    print("Reading documents")
    f = open(file_path, "r")
    print("Read done")
    # iter = file_iterator_interval(f, 0, 100)
    iter = f
    output: List[Tuple[str, SegmentwiseTokenizedText]] = jsonl_to_swtt(iter, get_tokenizer(), 20000)
    save_to_pickle(output, "ca_run3_swtt")


def size_check():
    print("Loading ca_run3_swtt")
    docs: List[Tuple[str, SegmentwiseTokenizedText]] = load_from_pickle("ca_run3_swtt")
    window_size = 400
    skip = 0
    window_enum_policy = WindowEnumPolicy(skip)
    averager = Averager()
    print("start enum")
    for doc_id, doc in docs:
        n_window = len(window_enum_policy.window_enum(doc, window_size))
        averager.append(n_window)

    print("Total {} runs, avg {}".format(sum(averager.history), averager.get_average()))


if __name__ == "__main__":
    size_check()
