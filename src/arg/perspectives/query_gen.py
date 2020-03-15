import os

from arg.perspectives.basic_analysis import get_candidates
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids
from cpath import output_path
from misc_lib import lmap, exist_or_mkdir
from tlm.retrieve_lm.galago_query_maker import clean_query, get_query_entry_bm25_anseri, save_queries_to_file


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
    d_ids = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    all_data_points = get_candidates(claims)

    def get_query_entry_from_data_point(data_point):
        label, cid, pid, claim_text, p_text = data_point
        tokens = claim_text.split() + p_text.split()
        query_text = clean_query(tokens)
        qid = cid + "_" + pid
        return get_query_entry_bm25_anseri(qid, query_text)

    queries = lmap(get_query_entry_from_data_point, all_data_points)

    out_dir = os.path.join(output_path, "perspective_train_claim_perspective_query")
    exist_or_mkdir(out_dir)
    n_query_per_file = 50

    i = 0
    while i * n_query_per_file < len(queries):
        st = i * n_query_per_file
        ed = (i+1) * n_query_per_file
        out_path = os.path.join(out_dir, "{}.json".format(i))
        save_queries_to_file(queries[st:ed], out_path)
        i+= 1


if __name__ == "__main__":
    write_claim_perspective_pair_as_query()