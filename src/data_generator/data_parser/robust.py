import path
import os
import data_generator.data_parser.trec as trec
from data_generator.data_parser.trec import load_trec
robust_path = os.path.join(path.data_path, "robust")

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


def load_robust04_query():
    path = os.path.join(robust_path, "topics.txt")
    queries = load_trec(path, 2)
    for key in queries:
        queries[key] = queries[key].strip()
    return queries


def load_2k_rank():
    path =os.path.join(robust_path, "rob04.desc.galago.2k.out")
    f = open(path, "r")

    ranked_list = {}

    for line in f:
        q_id, _, doc_id, rank, score, _ = line.split()

        if q_id not in ranked_list:
            ranked_list[q_id] = []

        ranked_list[q_id].append((doc_id, int(rank), score))

    return ranked_list


def sanity_check():
    ranked_list = load_2k_rank()
    collection = trec.load_robust(trec.robust_path)
    def process(doc):
        return doc.lower().split()
    for q_id, listings in ranked_list.items():
        for doc_id, rank, score in listings[:1]:
            docs_path = os.path.join(path.data_path, "robust", "docs", doc_id)
            content = process(collection[doc_id])
            open(docs_path, "w").write(content)



if __name__ == '__main__':
    queries = load_robust04_query()
    for q_id in queries:
        print("\t".join([q_id] + queries[q_id].split()))
    #sanity_check()


