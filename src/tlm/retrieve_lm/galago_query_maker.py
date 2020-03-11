import json
import string

import cpath
from cache import *


def clean_query(query):
    q_term = []
    spe_chars = set([t for t in string.printable if not t.isalnum()])
    for t in query:
        if t in spe_chars:
            continue
        else:
            q_term.append(t)
    return q_term


def get_query_entry(q_id, query):
    return {"number": str(q_id), "text": "#combine({})".format(" ".join(query))}


def get_query_entry_bm25_anseri(q_id, query):
    return {"number": str(q_id), "text": "#combine(bm25:K=0.9:b=0.4({}))".format(" ".join(query))}


def save_queries_to_file(queries, out_path):
    data = {"queries": queries}
    fout = open(out_path, "w")
    fout.write(json.dumps(data, indent=True))
    fout.close()


def main():
    print("Start")
    spr = StreamPickleReader("robust_candi_query_")
    query_per_task = 1000 * 10
    out_idx = 0
    while spr.has_next():
        queries = []
        for i in range(query_per_task):
            if not spr.has_next():
                break
            q_id, query = spr.get_item()
            query = clean_query(query)
            queries.append(get_query_entry(q_id, query))

        out_path = os.path.join(cpath.output_path, "query", "g_query_{}.json".format(out_idx))
        save_queries_to_file(queries, out_path)
        out_idx += 1


if __name__ == "__main__":
    main()
