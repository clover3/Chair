from typing import List

from arg.counter_arg.eval import load_problems
from arg.counter_arg.header import splits, ArguDataPoint
from arg.qck.decl import QCKQuery, QKUnit
from arg.qck.kd_candidate_gen import QKWorker
from base_type import FilePath
from cache import save_to_pickle
from list_lib import lmap
from misc_lib import TimeEstimator


def config1():
    return {
        'step_size': 300,
        'window_size': 300,
        'top_n': 10,
    }


def config2():
    return {
        'step_size': 30,
        'window_size': 300,
        'top_n': 10,
    }


def generate_for_all_split():
    for split in splits[1:]:
        generate(split)


#aa
def generate(split):
    print("Generate for ", split)
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/counter_arg/q_res/{}_all.txt".format(split))
    candidate = get_candidates(q_res_path, split, config1())
    print("Num candidate : {}", len(candidate))
    save_to_pickle(candidate, "ca_qk_candidate_{}".format(split))


def get_candidates(q_res_path, split, config) -> List[QKUnit]:
    problems: List[ArguDataPoint] = load_problems(split)
    top_n = config['top_n']
    print("{} problems".format(len(problems)))

    def problem_to_qckquery(problem: ArguDataPoint):
        return QCKQuery(str(problem.text1.id.id), problem.text1.text)

    print("Making queries")
    queries: List[QCKQuery] = lmap(problem_to_qckquery, problems)

    worker = QKWorker(q_res_path, config, top_n)
    all_candidate = []
    ticker = TimeEstimator(len(queries))
    for q in queries:
        ticker.tick()
        try:
            doc_part_list = worker.work(q)
            e = q, doc_part_list
            all_candidate.append(e)
        except KeyError as e:
            print(e)
    return all_candidate


if __name__ == "__main__":
    generate_for_all_split()
