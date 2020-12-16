from typing import List, Dict

from arg.qck.decl import QKUnit
from arg.qck.kd_candidate_gen import get_qk_candidate
from arg.robust.qc_common import to_qck_queries
from arg.robust.runner.robust_on_robust_qk_gen import config1
from base_type import FilePath
from cache import save_to_pickle
from data_generator.data_parser.robust import load_robust04_title_query


def get_candidates(q_res_path, config) -> List[QKUnit]:
    queries: Dict[str, str] = load_robust04_title_query()
    print("{} queries".format(len(queries)))
    qck_queries = to_qck_queries(queries)

    return get_qk_candidate(config, q_res_path, qck_queries)


def main():
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/robust_on_clueweb/robust_clue_query_res.txt")
    candidate = get_candidates(q_res_path, config1())
    print("Num candidate : {}", len(candidate))
    save_to_pickle(candidate, "robust_on_clueweb_qk_candidate")


if __name__ == "__main__":
    main()

