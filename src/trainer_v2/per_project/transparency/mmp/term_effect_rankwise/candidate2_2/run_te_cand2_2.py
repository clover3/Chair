import logging
import sys

from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_2_path_helper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.te_with_configs import run_term_effect_w_path_helper


def main():
    path_helper = get_cand2_2_path_helper()
    c_log.setLevel(logging.DEBUG)
    st = int(sys.argv[1])
    ed = int(sys.argv[2])
    job_name = f"run_candidate_{st}_{ed}"
    with JobContext(job_name):
        run_term_effect_w_path_helper(path_helper, st, ed)


if __name__ == "__main__":
    main()
