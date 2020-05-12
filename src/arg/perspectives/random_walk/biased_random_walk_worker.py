import os
import pickle
from collections import Counter
from typing import List, Dict

from arg.perspectives.random_walk.biased_random_walk import run_biased_random_walk
from arg.perspectives.random_walk.random_walk_worker import select_vertices_edges
from cache import load_pickle_from
from data_generator import job_runner


class BiasedRandomWalkWorker(job_runner.WorkerInterface):
    def __init__(self, out_path, input_dir, prob_score_d):
        self.out_dir = out_path
        self.input_dir = input_dir
        self.prob_score_d: Dict[int, List] = prob_score_d
        self.p_reset = 0.1
        self.max_repeat = 1000

    def work(self, job_id):
        file_no = int(job_id / 10)
        idx = job_id % 10
        pc_co_occurrence = load_pickle_from(os.path.join(self.input_dir, str(file_no)))
        cid, pair_counter = pc_co_occurrence[idx]
        edges, valid_vertices = select_vertices_edges(pair_counter)
        try:
            init_p_dict = Counter(dict(self.prob_score_d[cid]))
            result = run_biased_random_walk(edges, valid_vertices, self.max_repeat, self.p_reset, init_p_dict)
            result = Counter(result)
            output = cid, result
            save_path = os.path.join(self.out_dir, str(job_id))
            pickle.dump(output, open(save_path, "wb"))
        except KeyError as e:
            print(e)


