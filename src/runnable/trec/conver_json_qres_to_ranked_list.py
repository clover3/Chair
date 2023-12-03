import json
import sys

from trec.ranked_list_util import json_qres_to_ranked_list
from trec.trec_parse import write_trec_ranked_list_entry


def main():
    q_res_path = sys.argv[1]
    ranked_list_save_path = sys.argv[2]
    run_name = sys.argv[3]

    j = json.load(open(q_res_path, "r"))
    tr_entries = json_qres_to_ranked_list(j, run_name)
    write_trec_ranked_list_entry(tr_entries, ranked_list_save_path)


if __name__ == "__main__":
    main()
