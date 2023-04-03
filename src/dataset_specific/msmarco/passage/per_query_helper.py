import os

from cpath import output_path, data_path
from misc_lib import path_join



def get_root_dir():
    root_dir = path_join(data_path, "msmarco", "passage", "grouped_1000")
    return root_dir


def get_per_query_doc_path(query_id):
    group_id = int(int(query_id) / 1000)



