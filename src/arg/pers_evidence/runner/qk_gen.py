from arg.pers_evidence.common import get_qck_queries_all
from arg.qck.kd_candidate_gen import get_qk_candidate
from arg.robust.runner.robust_on_robust_qk_gen import config1
from base_type import FilePath
from cache import save_to_pickle


def main():
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/qck/evidence/q_res_10.txt")
    qck_queries = get_qck_queries_all()
    candidate = get_qk_candidate(config1(), q_res_path, qck_queries)
    print("Num candidate : {}", len(candidate))
    save_to_pickle(candidate, "pc_evidence_qk")


if __name__ == "__main__":
    main()

