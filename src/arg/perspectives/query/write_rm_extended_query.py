import os
from typing import List, Dict, Tuple

from cpath import output_path
from galagos.interface import DocQuery
from galagos.parse import load_queries, save_queries_to_file
from galagos.tokenize_util import drop_words
from list_lib import left
from models.classic.stopword import load_stopwords_for_query


def extract_terms_from_structured_query(text: str):
    combine_str = "#combine("
    bm25 = "#bm25:K=0("
    text = text.strip()

    assert text.startswith(combine_str)
    text = text[len(combine_str):-1]

    tokens = text.split()
    out_terms = []
    for t in tokens:
        t = t.strip()
        assert t.startswith(bm25)
        t = t[len(bm25):-1]
        out_terms.append(t)
    return out_terms


def get_ex_terms(ex_info_dir, qid) -> List[Tuple[str, float]]:
    file_path = os.path.join(ex_info_dir, qid)
    output = []
    n_err = 0
    for line in open(file_path, "r"):
        try:
            tokens = line.split()
            if len(tokens) > 2:
                weight = weight[-1]
                for t in tokens:
                    output.append((t, float(weight)))
            elif len(tokens) == 2:
                term, weight = tokens
                output.append((term, float(weight)))
            else:
                raise ValueError

        except ValueError:
            n_err += 1
            print(line)
##
    if n_err > 1:
        print(n_err)
    return output


def format_query_bm25(query_id: str,
                      terms: List[str],
                      weights_d: Dict[str, float],
                      k=0,
                      ) -> DocQuery:
    weight_list = [weights_d[t] for t in terms]

    combine_weight = ":".join(["{0}={1:.6f}".format(idx, w) for idx, w in enumerate(weight_list)])
    q_str_inner = " ".join(["#bm25:K={}({})".format(k, t) for t in terms])
    query_str = "#combine:{}({})".format(combine_weight, q_str_inner)
    return DocQuery({
        'number': query_id,
        'text': query_str
    })


def drop_dots(tokens):
    return list([t for t in tokens if "." not in t])


def main():
    split = "dev"
    stopword = load_stopwords_for_query()
    # split = "train"
    ex_info_dir = "/mnt/nfs/work3/youngwookim/job_man/pc_rm_terms_{}".format(split)
    query_path = os.path.join(output_path, "perspective_{}_claim_query_k0_fixed.json".format(split))
    queries = load_queries(query_path)
    ex_w_scale = 100
    out_path = os.path.join(output_path, "perspective_query",
                            "pc_{}_claim_query_rm_ex.json".format(split))
##
    new_queries = get_extended(ex_info_dir, ex_w_scale, queries, stopword)
    save_queries_to_file(new_queries, out_path)


def get_extended(ex_info_dir, ex_w_scale, queries, stopword):
    new_queries = []
    for q in queries:
        qid = q['number']
        text = q['text']
        q_terms: List[str] = extract_terms_from_structured_query(text)
        ex_terms_and_weight: List[Tuple[str, float]] = get_ex_terms(ex_info_dir, qid)
        ex_terms = left(ex_terms_and_weight)
        weights_d = dict(ex_terms_and_weight)
        weights_d = {k: w * ex_w_scale for k, w in weights_d.items()}

        q_terms = drop_words(q_terms, stopword)
        ex_terms = drop_dots(drop_words(ex_terms, stopword))
        all_terms = q_terms + ex_terms
        q_term_weight = 1 / len(q_terms)
        for term in q_terms:
            if term not in weights_d:
                weights_d[term] = q_term_weight
            else:
                weights_d[term] += q_term_weight
        doc_query = format_query_bm25(qid, all_terms, weights_d)
        new_queries.append(doc_query)
    return new_queries


if __name__ == "__main__":
    main()