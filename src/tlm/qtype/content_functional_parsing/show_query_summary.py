import os
from typing import Dict

from cache import load_from_pickle
from cpath import output_path
from misc_lib import group_by
from tab_print import save_table_as_csv
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import load_query_info_dict


def main():
    split = "train"
    qtype_info_dict = load_query_info_dict(split)
    n_all_query = len(qtype_info_dict)
    qtype_id_mapping: Dict[str, int] = load_from_pickle("qtype_id_mapping")

    def get_qtype_id(info):
        return qtype_id_mapping[" ".join(info.functional_tokens)]

    grouped = group_by(qtype_info_dict.values(), get_qtype_id)

    rows = []
    for qtype_id in range(1, 2048):
        queries_with_qtype = grouped[qtype_id]
        some_query = queries_with_qtype[0]
        rep = " ".join(some_query.out_s_list)
        n_query = len(queries_with_qtype)
        row = [qtype_id,
               " ".join(some_query.functional_tokens),
               n_query,
               n_query / n_all_query,
               rep]
        rows.append(row)

    save_path = os.path.join(output_path, "qtype", "qtype_summary.csv")
    save_table_as_csv(rows, save_path)


    # demo_parsing("train")
    # return demo_frequency("train")
    # save_qtype_id()


if __name__ == "__main__":
    main()
