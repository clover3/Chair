import os

from arg.perspectives.load import splits
from arg.perspectives.qck.qck_common import get_qck_queries
from arg.perspectives.runner_qck.run_qk_candidate_gen import config1
from arg.qck.kd_candidate_gen import get_qk_candidate
from cache import save_to_pickle
from cpath import output_path


def main():
    for split in splits:
        q_res_path = os.path.join(output_path,
                                  "perspective_experiments",
                                  "clueweb_qres",
                                  "{}.txt".format(split))
        qck_queries = get_qck_queries(split)
        candidate = get_qk_candidate(config1(), q_res_path, qck_queries)
        print("Num candidate : {}", len(candidate))
        save_to_pickle(candidate, "pc_qk2_{}".format(split))


if __name__ == "__main__":
    main()

