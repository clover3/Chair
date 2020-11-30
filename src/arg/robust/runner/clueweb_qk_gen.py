from typing import List, Dict

from arg.qck.decl import QKUnit
from arg.qck.kd_candidate_gen import QKWorker
from arg.robust.qc_common import to_qck_queries
from arg.robust.runner.robust_on_robust_qk_gen import config1
from base_type import FilePath
from cache import save_to_pickle
from data_generator.data_parser.robust import load_robust04_query
from misc_lib import TimeEstimator


def get_candidates(q_res_path, config) -> List[QKUnit]:
    top_n = config['top_n']
    queries: Dict[str, str] = load_robust04_query()
    print("{} queries".format(len(queries)))
    qck_queries = to_qck_queries(queries)

    worker = QKWorker(q_res_path, config, top_n)
    all_candidate = []
    ticker = TimeEstimator(len(queries))
    for q in qck_queries:
        ticker.tick()
        try:
            doc_part_list = worker.work(q)
            e = q, doc_part_list
            all_candidate.append(e)
        except KeyError as e:
            print(e)
    return all_candidate


def main():
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/robust_on_clueweb/robust_clue_query_res.txt")
    candidate = get_candidates(q_res_path, config1())
    print("Num candidate : {}", len(candidate))
    save_to_pickle(candidate, "robust_on_clueweb_qk_candidate")


if __name__ == "__main__":
    main()

