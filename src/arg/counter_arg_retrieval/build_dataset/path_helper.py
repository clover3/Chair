import os

from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped


def load_sliced_passage_ranked_list(run_name):
    save_path = os.path.join(output_path, "ca_building",
                             "passage_ranked_list_sliced", '{}.txt'.format(run_name))
    return load_ranked_list_grouped(save_path)