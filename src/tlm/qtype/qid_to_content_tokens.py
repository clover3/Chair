from collections import Counter
from typing import List, Dict

from cache import load_from_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer


def get_qid_to_content_tokens(split):
    obj = load_from_pickle("mmd_query_parse_{}".format(split))
    parsed_queries: List[Dict] = obj[0]
    func_tokens_counter: Counter = obj[1]
    tokenizer = get_tokenizer()
    out_dict = {}
    for d in parsed_queries:
        content_spans = d['content_span']
        content_tokens_bert_tokenized = tokenizer.tokenize(content_spans)
        out_dict[d['qid']] = content_tokens_bert_tokenized

    return out_dict


def demo_frequency(split):
    obj = load_from_pickle("mmd_query_parse_{}".format(split))
    parsed_queries: List[Dict] = obj[0]
    func_tokens_counter: Counter = obj[1]
    n_minor_query = 0
    for d in parsed_queries:
        func_str = " ".join(d['functional_tokens'])
        if func_tokens_counter[func_str] < 10:
            n_minor_query += 1
            # print("----")
            # print(d['query'], func_tokens_counter[func_str])
            # print(" ".join(d['out_s_list']))
    print("{0:.2f} are minor ({1}/{2})".format(n_minor_query / len(parsed_queries), n_minor_query, len(parsed_queries)))


def main():
    return demo_frequency("train")


if __name__ == "__main__":
    main()
