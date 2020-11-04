import os
from typing import List, Dict, Tuple


def load_doc_scores(dir_path, max_job) -> Dict[int, List[Tuple[str, float]]]:
    out_d: Dict[int, List] = {}
    for i in range(max_job):
        path = os.path.join(dir_path, str(i))
        entries = []
        for line in open(path, "r"):
            doc_id, score = line.split()
            score = float(score)
            entries.append((doc_id, score))
        out_d[i] = entries
#
    return out_d