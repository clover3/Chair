import os
from typing import List

from arg.perspectives.basic_analysis import get_candidates
from arg.perspectives.declaration import PerspectiveCandidate
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids, load_dev_claim_ids, load_test_claim_ids, \
    splits, load_claim_ids_for_split
from arg.perspectives.pc_run_path import query_dir_format
from cpath import output_path, pjoin
from galagos.interface import format_query_bm25, DocQuery, write_queries_to_files
from galagos.parse import clean_query, get_query_entry_bm25_anseri, save_queries_to_file
from galagos.tokenize_util import clean_tokenize_str_to_tokens
from list_lib import lmap
from misc_lib import exist_or_mkdir
from models.classic.stopword import load_stopwords


def write_claim_as_query():
    d_ids = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    queries = []
    for c in claims:
        cid = c["cId"]
        claim_text = c["text"]
        tokens = claim_text.split()
        query_text = clean_query(tokens)
        print(query_text)
        q_entry = get_query_entry_bm25_anseri(cid, query_text)
        queries.append(q_entry)

    out_path = os.path.join(output_path, "perspective_dev_claim_query.json")
    save_queries_to_file(queries, out_path)


def get_claims_as_plain_query(claims):
    q_str_list = []
    for query in get_claims_query(claims):
        q_str = query['text']
        q_str_list.append(q_str)
    return q_str_list


def get_claims_query(claims, drop_stopwords=False):
    if drop_stopwords:
        stopword = load_stopwords()

    queries = []
    for c in claims:
        cid = str(c["cId"])
        claim_text = c["text"]
        q_terms: List[str] = clean_tokenize_str_to_tokens(claim_text)
        print(q_terms)
        if drop_stopwords:
            q_terms = list([t for t in q_terms if t not in stopword])
        q_terms = list([t.replace(".", "") for t in q_terms])
        print(q_terms)

        q_entry = format_query_bm25(cid, q_terms)
        queries.append(q_entry)
    return queries


def run_write_claims_as_plain_query():
    for claim_ids, out_name in [(load_train_claim_ids(), "train_claim_query_raw.txt"),
                                (load_dev_claim_ids(), "dev_claim_query_raw.txt")]:
        claims = get_claims_from_ids(claim_ids)
        q_str_list = get_claims_as_plain_query(claims)
        f = open(pjoin(output_path, out_name), "w")
        for s in q_str_list:
            f.write(s + "\n")


def write_claim_queries_k0():
    def write(claim_ids, split_name):
        claims = get_claims_from_ids(claim_ids)
        queries = get_claims_query(claims, True)
        out_path = os.path.join(output_path, "perspective_{}_claim_query_k0.json".format(split_name))
        save_queries_to_file(queries, out_path)

    claim_ids, split_name = (load_train_claim_ids(), "train")

    write(claim_ids, split_name)
    claim_ids, split_name = (load_dev_claim_ids(), "dev")
    write(claim_ids, split_name)


def write_claim_queries2():
    for split in splits:
        claim_ids = load_claim_ids_for_split(split)
        claims = get_claims_from_ids(claim_ids)
        queries = get_claims_query(claims, True)
        out_path = os.path.join(output_path, "perspective_claim_query2_{}.json".format(split))
        save_queries_to_file(queries, out_path)


def write_claim_perspective_pair_as_query():
    split = "dev"
    assert split in ["train", "dev", "test"]

    d_ids = list({
        "train": load_train_claim_ids(),
        "dev": load_dev_claim_ids(),
        "test": load_test_claim_ids()
    }[split])
    claims = get_claims_from_ids(d_ids)
    print(len(claims), " claims")
    is_train = split == "train"
    all_data_points = get_candidates(claims, is_train)
    k = 0

    def get_query_entry_from_data_point(x : PerspectiveCandidate) -> DocQuery:
        tokens = clean_tokenize_str_to_tokens(x.claim_text + " " + x.p_text)
        qid = "{}_{}".format(x.cid, x.pid)
        return format_query_bm25(qid, tokens, k)

    queries = lmap(get_query_entry_from_data_point, all_data_points)

    out_dir = query_dir_format.format(split)
    exist_or_mkdir(out_dir)
    n_query_per_file = 50

    write_queries_to_files(n_query_per_file, out_dir, queries)



if __name__ == "__main__":
    write_claim_queries2()

