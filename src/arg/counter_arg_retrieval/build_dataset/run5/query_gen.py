import json
import os

from arg.counter_arg_retrieval.build_dataset.data_prep.pc_query_gen import generate_query, make_galago_query
from arg.counter_arg_retrieval.build_dataset.run5.data_path_common import get_run5_query_set
from cpath import output_path


def main():
    ca_task_list = get_run5_query_set()
    qid_list = [ca_task.qid for ca_task in ca_task_list]
    queries = generate_query(qid_list)
    g_queries = make_galago_query(queries)
    out_path = os.path.join(output_path, "ca_building", "run5", "queries.json")
    fout = open(out_path, "w")
    fout.write(json.dumps(g_queries, indent=True))
    fout.close()


if __name__ == "__main__":
    main()
