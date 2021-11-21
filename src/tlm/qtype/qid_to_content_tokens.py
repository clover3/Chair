from collections import Counter
from typing import List, Dict

from cache import load_from_pickle, save_to_pickle
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


def func_tokens_to_qtype_id(split):
    obj = load_from_pickle("mmd_query_parse_{}".format(split))
    parsed_queries: List[Dict] = obj[0]
    func_tokens_counter: Counter = obj[1]

    qtype_id = 1
    qtype_d = {}
    for func_tokens, cnt in func_tokens_counter.most_common():
        qtype_d[func_tokens] = qtype_id
        qtype_id += 1

    print("{} qtype ids".format(qtype_id))
    return qtype_d


def save_qtype_id():
    qtype_id = func_tokens_to_qtype_id("train")
    save_to_pickle(qtype_id, "qtype_id_mapping")


def get_qid_to_qtype_id(split) -> Dict[str, int]:
    qtype_id_mapping: Dict[str, int] = load_from_pickle("qtype_id_mapping")
    obj = load_from_pickle("mmd_query_parse_{}".format(split))
    parsed_queries: List[Dict] = obj[0]
    func_tokens_counter: Counter = obj[1]
    out_d = {}
    for d in parsed_queries:
        func_str = " ".join(d['functional_tokens'])
        try:
            qtype_id = qtype_id_mapping[func_str]
        except KeyError:
            qtype_id = 0
        out_d[d['qid']] = qtype_id
    return out_d


def demo_frequency(split):
    obj = load_from_pickle("mmd_query_parse_{}".format(split))
    parsed_queries: List[Dict] = obj[0]
    func_tokens_counter: Counter = obj[1]
    n_minor_query = 0


    n_category_major_only = 0
    for func_str, cnt in func_tokens_counter.items():
        if cnt > 10:
            n_category_major_only += 1

    print("Num queries: ", len(parsed_queries))
    print("{} categories".format(n_category_major_only))
    for d in parsed_queries:
        func_str = " ".join(d['functional_tokens'])
        if func_tokens_counter[func_str] < 10:
            n_minor_query += 1
            # print("----")
            # print(d['query'], func_tokens_counter[func_str])
            # print(" ".join(d['out_s_list']))
    print("{0:.2f} are minor ({1}/{2})".format(n_minor_query / len(parsed_queries), n_minor_query, len(parsed_queries)))


def main():
    # return demo_frequency("train")
    save_qtype_id()


if __name__ == "__main__":
    main()
