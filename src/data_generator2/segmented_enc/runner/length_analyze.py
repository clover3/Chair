from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.mnli.mnli_reader import MNLIReader
from dataset_specific.mnli.sci_tail import SciTailReaderTFDS
from dataset_specific.mnli.snli_reader_tfds import SNLIReaderTFDS


def counter_summary(counter: Counter, intervals: List[int]):
    n_item = sum(counter.values())
    less_than = Counter()
    for i in range(max(intervals) + 1):
        less_than[i] = counter[i] + less_than[i - 1]

    for i in intervals:
        proportion = less_than[i] / n_item
        print(f"< {i}: {proportion:.5f}")


def do_length_analyze(nli_reader):
    tokenizer = get_tokenizer()
    def get_n_token(text):
        return len(tokenizer.tokenize(text))

    counter = Counter()
    for item in nli_reader.get_dev():
        n_p = get_n_token(item.premise)
        n_h = get_n_token(item.hypothesis)
        n = n_p + n_h
        counter[n] += 1

    interval = [50, 100, 150, 200, 300]
    counter_summary(counter, interval)


def main():
    reader = MNLIReader()
    do_length_analyze(reader)


if __name__ == "__main__":
    main()