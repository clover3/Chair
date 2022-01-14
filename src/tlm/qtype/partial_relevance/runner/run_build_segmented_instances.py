import os
import pickle

from cpath import output_path
from epath import job_man_dir
from tlm.qtype.partial_relevance.problem_builder import build_eval_instances


def main():
    info_path = os.path.join(job_man_dir, "MMDE_dev_info")
    raw_prediction_path = os.path.join(output_path, "qtype", "MMDE_dev_mmd_Z.score")
    items = build_eval_instances(info_path, raw_prediction_path, 10)
    save_path = os.path.join(output_path, "qtype", "MMDE_dev_problems.pickle")
    pickle.dump(items, open(save_path, "wb"))


if __name__ == "__main__":
    main()
