import csv
import os
from collections import Counter, defaultdict
from typing import List, Dict, NamedTuple, Tuple

from cache import load_from_pickle, save_to_pickle, named_tuple_to_json
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from trec.types import QueryID


class QueryInfo(NamedTuple):
    qid: QueryID
    query: str
    content_span: str
    functional_tokens: List[str]
    out_s_list: List[str]

    def get_head_tail(self):
        head, body, tail = self.get_head_body_tail()
        return head, tail

    def get_head_body_tail(self):
        state = "head"
        head = []
        tail = []
        body = []
        for token in self.out_s_list:
            if token == "[":
                state = "middle"
            elif token == "]":
                state = "tail"
            else:
                if state == "head":
                    head.append(str(token))
                elif state == "tail":
                    tail.append(str(token))
                elif state == "middle":
                    body.append(str(token))
        return head, body, tail

    def get_func_span_rep(self):
        return self.query.replace(self.content_span, "[MASK]", 1)

    def get_q_seg_indices(self):
        head, body, tail = self.get_head_body_tail()
        seg2_start = len(head)
        seg1_tail_start = seg2_start + len(body)

        seg1_head_indice = [i for i, _ in enumerate(head)]
        seg1_tail_indice = [i + seg1_tail_start for i, _ in enumerate(tail)]
        q_seg1_indice = seg1_head_indice + seg1_tail_indice
        seg2_indices = [i + seg2_start for i, _ in enumerate(body)]
        q_seg_indices = [q_seg1_indice, seg2_indices]
        return q_seg_indices

    def to_json(self):
        return named_tuple_to_json(self)

    @classmethod
    def from_json(cls, j):
        return QueryInfo(j['qid'], j['query'], j['content_span'], j['functional_tokens'], j['out_s_list'])


def get_qid_to_content_tokens(split) -> Dict[QueryID, List[str]]:
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


def load_query_info_dict(split) -> Dict[str, QueryInfo]:
    obj = load_from_pickle("mmd_query_parse_{}".format(split))
    parsed_queries: List[Dict] = obj[0]
    out_d = {}
    for d in parsed_queries:
        qi = QueryInfo(d['qid'], d['query'], d['content_span'], d['functional_tokens'], d['out_s_list'])
        out_d[qi.qid] = qi
    return out_d


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
            print("----")
            print(d['query'], func_tokens_counter[func_str])
            print(" ".join(d['out_s_list']))
    print("{0:.2f} are minor ({1}/{2})".format(n_minor_query / len(parsed_queries), n_minor_query, len(parsed_queries)))


def demo_parsing(split):
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
        print("----")
        print(d['query'], func_tokens_counter[func_str])
        print(" ".join(d['out_s_list']))
    print("{0:.2f} are minor ({1}/{2})".format(n_minor_query / len(parsed_queries), n_minor_query, len(parsed_queries)))


def print_qtype():
    qtype_id_mapping: Dict[str, int] = load_from_pickle("qtype_id_mapping")
    part = list(qtype_id_mapping.items())[:2048]
    f = open(os.path.join(output_path, "qtype", "../qtype.csv"), "w", newline="")
    writer = csv.writer(f)
    for text, qtype_id in part:
        writer.writerow([text, str(qtype_id)])


def structured_qtype_text(query_info_dict: Dict[str, QueryInfo]) -> Dict[str, Tuple[str, str]]:
    mapping_counter = defaultdict(Counter)
    for e in query_info_dict.values():
        st = e.out_s_list.index("[")
        ed = e.out_s_list.index("]")
        func_rep = " ".join(e.functional_tokens)
        head = " ".join(e.out_s_list[:st])
        tail = " ".join(e.out_s_list[ed+1:])
        mapping_counter[func_rep][(head, tail)] += 1

    mapping = {}
    for func_rep, counter in mapping_counter.items():
        (head, tail), cnt = counter.most_common(1)[0]
        mapping[func_rep] = (head, tail)
    return mapping


def main():
    print_qtype()
    # demo_parsing("train")
    # return demo_frequency("train")
    # save_qtype_id()


if __name__ == "__main__":
    main()
