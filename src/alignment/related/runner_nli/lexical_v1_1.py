import functools
import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set

from alignment import RelatedEvalInstance
from alignment.data_structure.eval_data_structure import RelatedEvalAnswer, RelatedBinaryAnswer, join_a_p
from alignment.nli_align_path_helper import load_mnli_rei_problem
from alignment.related.related_answer_data_path_helper import load_related_eval_answer, \
    get_related_binary_save_path, save_json_at
from data_generator.tokenizer_wo_tf import get_tokenizer
from models.classic.stopword import load_stopwords


def get_is_stopword_fn():
    stopwords = load_stopwords()
    tokenizer = get_tokenizer()
    ids_stopword = []
    for word in stopwords:
        tokens = tokenizer.tokenize(word)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) > 1:
            print("Word {} has {} as ids".format(word, ids))
        else:
            ids_stopword.append(ids[0])

    def is_stopword(tokens):
        if len(tokens) > 1:
            return False
        return tokens[0] in ids_stopword

    return is_stopword


def get_convert_answer_fn(threshold):
    is_stopword = get_is_stopword_fn()

    def convert_answer(a: RelatedEvalAnswer, p: RelatedEvalInstance) -> RelatedBinaryAnswer:
        def convert_value(s):
            if s >= threshold:
                return 1
            else:
                return 0

        alignment: List[List[float]] = a.contribution.table
        new_alignment: List[List[int]] = []
        for seg_idx1, row in enumerate(alignment):
            tokens = p.seg_instance.text1.get_tokens_for_seg(seg_idx1)
            if is_stopword(tokens):
                new_row = [0 for _ in row]
            else:
                new_row = list(map(convert_value, row))
            new_alignment.append(new_row)

        return RelatedBinaryAnswer(a.problem_id, new_alignment)
    return convert_answer


def discretize_and_save(dataset_name, method, method_save):
    answers: List[RelatedEvalAnswer] = load_related_eval_answer(dataset_name, method)
    problem_list: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    a_p_list: List[Tuple[RelatedEvalAnswer, RelatedEvalInstance]] = join_a_p(answers, problem_list)

    convert_answer_fn = get_convert_answer_fn(0.7)
    new_answers: List[RelatedBinaryAnswer] = [convert_answer_fn(a, p) for a, p in  a_p_list]
    save_path = get_related_binary_save_path(dataset_name, method_save)
    save_json_at(new_answers, save_path)


def main():
    dataset = "train_head"
    discretize_and_save(dataset, "lexical_v1", "lexical_v1_1")


if __name__ == "__main__":
    main()
