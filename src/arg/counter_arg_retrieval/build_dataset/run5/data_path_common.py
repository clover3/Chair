import os
from typing import List

from arg.counter_arg_retrieval.build_dataset.ca_query import CATask
from cpath import output_path
from table_lib import read_csv_as_dict


def get_run5_query_set() -> List[CATask]:
    table_path = os.path.join(output_path, "ca_building", "run5", "query_set.csv")
    table = read_csv_as_dict(table_path)

    def encode_row(row):
        pc_stance = "Favor" if row['p_stance'] == row['c_stance'] else "Against"
        return CATask(row['qid'], row['claim'], row['c_stance'],
               row['perspective'], row['p_stance'],
               pc_stance, row['entity'])

    return list(map(encode_row, table))