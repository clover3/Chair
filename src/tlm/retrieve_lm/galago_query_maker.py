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
            queries.append({"number": str(q_id), "text": "#combine({})".format(" ".join(query))})

        data = {"queries": queries}

        out_path = os.path.join(cpath.output_path, "query", "g_query_{}.json".format(out_idx))
        fout = open(out_path, "w")
        fout.write(json.dumps(data, indent=True))
        fout.close()
        out_idx += 1


if __name__ == "__main__":
    main()
