import os
from typing import Dict

import cpath
import data_generator.data_parser.trec as trec
from data_generator.data_parser.robust2 import load_2k_rank, robust_path
from data_generator.data_parser.trec import load_trec
from trec.qrel_parse import load_qrels_structured


def load_robust04_query_krovetsz():
    q_path =os.path.join(robust_path, "rob04.desc.krovetz.txt")
    f = open(q_path, "r")
    result = []
    for line in f:
        tokens = line.split()
        q_id = tokens[0]
        q_terms = tokens[1:]
        result.append((q_id, q_terms))
    return result


def load_robust04_title_query() -> Dict[str, str]:
    path = os.path.join(robust_path, "topics.txt")
    queries = load_trec(path, 2)
    for key in queries:
        queries[key] = queries[key].strip()
    return queries


def load_robust04_qrels():
    path = os.path.join(robust_path, "qrels.rob04.txt")
    return load_qrels_structured(path)


def load_robust04_desc():
    q_path = os.path.join(robust_path, "04.testset")
    return parse_robust04_desc(q_path)


def load_robust04_desc2() -> Dict[str, str]:
    q_path = os.path.join(robust_path, "topics.robust04.txt")
    return parse_robust04_desc(q_path)


def parse_robust04_desc(q_path) -> Dict[str, str]:
    f = open(q_path, "r")
    STATE_DEF = 0
    STATE_DESC = 2
    state = STATE_DEF
    queries = dict()
    for line in f:
        if state == STATE_DEF:
            if line.startswith("<num>"):
                _, _, q_id = line.split()
            if line.startswith("<desc>"):
                state = STATE_DESC
                desc = ""
        elif state == STATE_DESC:
            if line.startswith("<narr>"):
                queries[q_id] = desc
                state = STATE_DEF
            else:
                desc += line
    return queries


def sanity_check():
    ranked_list = load_2k_rank()
    collection = trec.load_robust(trec.robust_path)
    def process(doc):
        return doc.lower().split()
    for q_id, listings in ranked_list.items():
        for doc_id, rank, score in listings[:1]:
            docs_path = os.path.join(cpath.data_path, "robust", "docs", doc_id)
            content = process(collection[doc_id])
            open(docs_path, "w").write(content)


def load_robust_04_query(query_type) -> Dict[str, str]:
    if query_type == "title":
        return load_robust04_title_query()
    elif query_type == "desc":
        return load_robust04_desc2()
    else:
        assert False


if __name__ == '__main__':
    #queries = load_robust04_query()
    queries = load_robust04_desc2()
    for q_id in queries:
        print("\t".join([q_id] + queries[q_id].split()))
    #sanity_check()
