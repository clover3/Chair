import os.path
import pickle
import sys
from collections import Counter
from typing import List, Iterable, Tuple

from misc_lib import path_join, TEL

from list_lib import left
from misc_lib import get_second
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import load_mmp_tfs, \
    read_deep_score_per_qid, load_qtfs
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import MMPGAlignConfig, \
    MMPGAlignPathHelper, get_mmp_galign_path_helper


def do_for_partition(split, partition_no):
    top_k_rel = 1
    tf = Counter()
    rel_count = Counter()

    qtfs: List[Tuple[str, Counter]] = load_qtfs(split, partition_no)
    for qid, query_tfs in TEL(qtfs):
        q_term_list: Iterable[str] = query_tfs.keys()
        qid_, score_entries = read_deep_score_per_qid(qid)
        score_entries.sort(key=get_second, reverse=True)
        rel_qids: List[str] = left(score_entries[:top_k_rel])
        qid, entries = load_mmp_tfs(qid)
        for doc_id, tfs in entries:
            is_rel = doc_id in rel_qids
            for d_term, cnt in tfs.items():
                tf[d_term] += 1
            if is_rel:
                for q_term in q_term_list:
                    for d_term, cnt in tfs.items():
                        key = q_term, d_term
                        rel_count[key] += cnt
    return tf, rel_count




def main():
    config: MMPGAlignPathHelper = get_mmp_galign_path_helper()
    partition_no = int(sys.argv[1])
    split = "train"
    pickle_save_path = config.get_sub_dir_partition_path("tf_rel_count", partition_no)
    if os.path.exists(pickle_save_path) and os.path.getsize(pickle_save_path) > 0:
        print("Skip ", partition_no)
    else:
        output = do_for_partition(split, partition_no)
        MB = 1000 * 1000
        count_size = sys.getsizeof(output)
        print("Object size", int(count_size / MB))
        pickle.dump(output, open(pickle_save_path, "wb"))



if __name__ == "__main__":
    main()
