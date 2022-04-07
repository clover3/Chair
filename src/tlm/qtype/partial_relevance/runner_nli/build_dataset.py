
#
import itertools

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.mnli.mnli_reader import MNLIReader
from tlm.qtype.partial_relevance.related_eval_instance import TextPair, get_word_level_rei
from typing import Iterable
from typing import List, Iterable, Callable, Dict, Tuple, Set


def build(split) -> List[TextPair]:
    tokenizer = get_tokenizer()
    for e in iter_mnli_dataset(split):
        rei = get_word_level_rei(e, tokenizer)
        yield rei


def iter_mnli_dataset(split) -> Iterable[TextPair]:
    reader = MNLIReader()
    for nli_pair in reader.load_split(split):
        yield TextPair(text_pair_id=nli_pair.pair_id,
                       query_like=nli_pair.hypothesis,
                       doc_like=nli_pair.premise
                       )


def main():
    num_items = 10000
    rei_iter = build("train")
    problems = itertools.islice(rei_iter, num_items)


    return NotImplemented


if __name__ == "__main__":
    main()

