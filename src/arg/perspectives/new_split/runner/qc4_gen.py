import os
from typing import Iterable

from arg.perspectives.load import splits
from arg.perspectives.new_split.common import get_qids_for_split, split_name2, \
    get_qck_candidate_for_split
from arg.perspectives.qck.qck_common import get_qck_queries_from_cids
from arg.perspectives.qck.qcknc_datagen import is_correct_factory
from arg.qck.qc.qc_common import make_pc_qc
from cpath import output_path
from list_lib import lmap
from misc_lib import exist_or_mkdir


def main():
    save_dir = os.path.join(output_path, "pc_qc4")
    exist_or_mkdir(save_dir)
    split_filename = split_name2
    for split in splits:
        qids: Iterable[str] = get_qids_for_split(split_filename, split)
        queries = get_qck_queries_from_cids(lmap(int, qids))
        eval_candidate = get_qck_candidate_for_split(split_filename, split)
        save_path = os.path.join(save_dir, split)
        make_pc_qc(queries, eval_candidate, is_correct_factory(), save_path)


if __name__ == "__main__":
    main()