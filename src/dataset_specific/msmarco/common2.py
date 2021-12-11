import os
from typing import List, Dict, Tuple

from epath import job_man_dir


def read_tokens_from_tsv(tsv_path):
    f = open(tsv_path, "r")
    out_d = {}
    for line in f:
        doc_id, url, title_tokens, body_tokens = line.split("\t")
        out_d[doc_id] = title_tokens, body_tokens
    return out_d


def load_token_d_for_job(split, job_id) -> Dict[str, Tuple[List[str], List[str]]]:
    save_path = os.path.join(job_man_dir, "MSMARCO_{}_tokens_tsv".format(split), job_id)
    return read_tokens_from_tsv(save_path)

