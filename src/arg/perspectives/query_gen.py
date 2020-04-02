import os

from arg.perspectives.basic_analysis import get_candidates, PerspectiveCandidate
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids, load_dev_claim_ids, load_test_claim_ids
from arg.perspectives.pc_run_path import query_dir_format
from cpath import output_path
from galagos.interface import format_query_bm25, DocQuery, write_queries_to_files
from galagos.parse import clean_query, get_query_entry_bm25_anseri, save_queries_to_file
from galagos.tokenize_util import clean_tokenize_str_to_tokens
from list_lib import lmap
from misc_lib import exist_or_mkdir


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
    all_data_points = get_candidates(claims)
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
    write_claim_perspective_pair_as_query()

