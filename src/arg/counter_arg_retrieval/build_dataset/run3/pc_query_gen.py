import json
import os

from arg.counter_arg_retrieval.build_dataset.data_prep.pc_query_gen import generate_query, make_galago_query
from cpath import output_path


def main():
    qid_list = ["47_351",
                "47_346",
                "47_1365",
                "504_4512",
                "504_3737",
                "504_24878",
                "504_4515",]

    queries = generate_query(qid_list)
    g_queries = make_galago_query(queries)
    out_path = os.path.join(output_path, "ca_building", "run3", "pc_queries.json")
    fout = open(out_path, "w")
    fout.write(json.dumps(g_queries, indent=True))
    fout.close()


if __name__ == "__main__":
    main()