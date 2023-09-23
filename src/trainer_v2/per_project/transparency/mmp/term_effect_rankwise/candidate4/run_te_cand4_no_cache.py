import logging
import sys

from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand4_path_helper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.te_measure_q_n_gram import run_te_config_wrap


def main():
    path_helper = get_cand4_path_helper()
    c_log.setLevel(logging.INFO)
    st = int(sys.argv[1])
    ed = int(sys.argv[2])
    job_name = f"run_candidate_{st}_{ed}"
    with JobContext(job_name):
        run_te_config_wrap(
            path_helper, st, ed,
            disable_cache=True,
        )


if __name__ == "__main__":
    main()
