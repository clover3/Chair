import json
import os
import sys
from collections import Counter
from typing import List, Dict

from arg.perspectives.load import load_claim_ids_for_split, load_claims_for_sub_split
from cpath import output_path
from galagos.interface import DocQuery
from galagos.parse import save_queries_to_file
from galagos.tokenize_util import clean_tokenize_str_to_tokens
from models.classic.stopword import load_stopwords_for_query


def load_terms(split) -> Dict[str, List[str]]:
    json_path = os.path.join(output_path, "cppnc", "token_scoring_{}.json".format(split))
    j = json.load(open(json_path, "r"))
    return j


def format_query_bm25(query_id: str,
                      main_q_terms: List[str],
                      aux_q_terms: List[str],
                      k=0,
                      aux_weight=0.1
                      ) -> DocQuery:

    def drop_dots(tokens):
        return list([t.replace(".", "") for t in tokens])

    main_q_terms = drop_dots(main_q_terms)
    aux_q_terms = drop_dots(aux_q_terms)

    all_terms = list()
    for t in main_q_terms:
        if t not in all_terms:
            all_terms.append(t)
    for t in aux_q_terms:
        if t not in all_terms:
            all_terms.append(t)
    weight = Counter()
    for t in main_q_terms:
        weight[t] += 1
    for t in aux_q_terms:
        weight[t] += aux_weight

    combine_weight = ":".join(["{}={}".format(idx, weight[t]) for idx, t in enumerate(all_terms)])
    q_str_inner = " ".join(["#bm25:K={}({})".format(k, t) for t in all_terms])
    query_str = "#combine:{}({})".format(combine_weight, q_str_inner)
    return DocQuery({
        'number': query_id,
        'text': query_str
    })


def drop_words(q_terms, words):
    q_terms = list([t for t in q_terms if t not in words])
    return q_terms


def get_claims_query(claims, expand_term_dict, drop_stopwords=False, aux_weight=0.1):
    if drop_stopwords:
        stopword = load_stopwords_for_query()

    queries = []
    for c in claims:
        cid = str(c["cId"])
        claim_text = c["text"]
        q_terms: List[str] = clean_tokenize_str_to_tokens(claim_text)
        ex_terms = expand_term_dict[cid]
        if drop_stopwords:
            q_terms = drop_words(q_terms, stopword)
            ex_terms = drop_words(ex_terms, stopword)

        q_entry = format_query_bm25(cid, q_terms, ex_terms, 0, aux_weight)
        queries.append(q_entry)
    return queries

# hello
def main():
    #split = "dev"
    aux_weight = float(sys.argv[1])
    split = "val"
    terms = load_terms(split)
    claim_ids = load_claim_ids_for_split(split)
    claims = load_claims_for_sub_split(split)
    #claims = get_claims_from_ids(claim_ids)
    queries = get_claims_query(claims, terms, True, aux_weight)
    out_path = os.path.join(output_path, "perspective_query", "perspective_{}_claim_query_extended.json".format(split))
    save_queries_to_file(queries, out_path)


if __name__ == "__main__":
    main()