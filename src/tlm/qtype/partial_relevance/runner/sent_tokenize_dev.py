from typing import List, Tuple

import nltk

from data_generator.tokenizer_wo_tf import pretty_tokens, get_tokenizer
from list_lib import flatten
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance
from tlm.qtype.partial_relevance.loader import load_dev_problems


def adhoc_fix(sents: List[str]) -> List[str]:
    # if a sentence is short and not first, prepend to previous sent
    tokenized_sents: List[List[str]] = [s.split() for s in sents]
    new_sents: List[List[str]] = []
    for i, s in enumerate(tokenized_sents):
        is_first = i == 0
        is_short = len(s) < 3
        non_single_char_tokens = [t for t in s if len(t) > 1]
        is_short = is_short or len(non_single_char_tokens) < 3
        f_prefend = not is_first and is_short

        if f_prefend:
            new_sents[-1].extend(s)
        else:
            new_sents.append(s)

    return [" ".join(s) for s in new_sents]


def sentence_segment(tokenizer, tokens_ids, debug_print=False) -> List[List[int]]:
    tokens = tokenizer.convert_ids_to_tokens(tokens_ids)
    s = pretty_tokens(tokens, True)
    if debug_print:
        print("---nltk tokenized---")
    sents = nltk.sent_tokenize(s)
    # for sent in sents:
    #     print(sent)
    sents: List[str] = adhoc_fix(sents)
    if debug_print:
        print("--- fixed ----")
        for sent in sents:
            print(sent)

    def to_ids(s: str) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))

    return [to_ids(s) for s in sents]


def sentence_segment_w_indices(tokenizer, tokens_ids, debug_print=False) -> Tuple[List[int], List[List[int]]]:
    sents: List[List[int]] = sentence_segment(tokenizer, tokens_ids, debug_print)
    flat_tokens = list(flatten(sents))
    loc = 0
    location_list = []
    for s in sents:
        st = loc
        ed = st + len(s)
        location_list.append(list(range(st, ed)))
        loc = ed
    return flat_tokens, location_list


def main():
    problems: List[RelatedEvalInstance] = load_dev_problems()
    tokenizer = get_tokenizer()
    for p in problems:
        tokens_ids = p.seg_instance.text2.tokens_ids
        sents = sentence_segment(tokenizer, tokens_ids, False)
        token_per_sent = len(tokens_ids) / len(sents)
        if token_per_sent < 10:
            print("WARNING sentences are short")
            sents = sentence_segment(tokenizer, tokens_ids, True)
        if len(sents) > 50:
            print("WARNING too many sentences")
            sents = sentence_segment(tokenizer, tokens_ids, True)


if __name__ == "__main__":
    main()