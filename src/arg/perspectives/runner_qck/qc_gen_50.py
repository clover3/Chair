import os

from arg.perspectives.load import splits
from arg.perspectives.qck.qck_common import get_qck_queries
from arg.perspectives.qck.qcknc_datagen import get_eval_candidates_as_qck, is_correct_factory
from arg.qck.qc.qc_common import make_pc_qc
from cpath import output_path
from misc_lib import exist_or_mkdir


def main():
    save_dir = os.path.join(output_path, "pc_qc")
    exist_or_mkdir(save_dir)
    for split in splits:
        queries = get_qck_queries(split)
        eval_candidate = get_eval_candidates_as_qck(split)
        save_path = os.path.join(save_dir, split)
        make_pc_qc(queries, eval_candidate, is_correct_factory(), save_path)


if __name__ == "__main__":
    main()
