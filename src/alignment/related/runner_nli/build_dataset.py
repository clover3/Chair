
#
import functools
from typing import Iterable, Callable

from alignment.data_structure.related_eval_instance import TextPair, get_word_level_rei, \
    RelatedEvalInstance
from alignment.nli_align_path_helper import get_rei_file_path
from cache import save_list_to_jsonl_w_fn
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.mnli.mnli_reader import MNLIReader


def build(split) -> Iterable[RelatedEvalInstance]:
    tokenizer = get_tokenizer()
    get_word_level_rei_fn: Callable[[TextPair], RelatedEvalInstance]\
        = functools.partial(get_word_level_rei, tokenizer)
    return map(get_word_level_rei_fn, iter_mnli_dataset(split))


def iter_mnli_dataset(split) -> Iterable[TextPair]:
    reader = MNLIReader()
    for nli_pair in reader.load_split(split):
        yield TextPair(text_pair_id=nli_pair.data_id,
                       query_like=nli_pair.hypothesis,
                       doc_like=nli_pair.premise
                       )


def main():
    split = "dev"
    rei_iter = build(split)
    problems = rei_iter
    save_path = get_rei_file_path(f"mnli_align_{split}.jsonl")
    save_list_to_jsonl_w_fn(problems, save_path, RelatedEvalInstance.to_json)


if __name__ == "__main__":
    main()

