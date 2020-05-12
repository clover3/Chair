import os
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Any

from cache import load_from_pickle
from data_generator import job_runner
from list_lib import left, lfilter
from summarization.random_walk import run_random_walk, Edges


def get_vertices_info(pair_counter) -> Counter:
    vertice_counter = Counter()
    for pair, cnt in pair_counter.items():
        a, b = pair
        vertice_counter[a] += cnt
        vertice_counter[b] += cnt

    return vertice_counter


def run_random_walk_wrap(max_repeat, p_reset, pair_counter) -> Counter:
    edges, valid_vertices = select_vertices_edges(pair_counter)
    result = run_random_walk(edges, valid_vertices, max_repeat, p_reset)
    result = Counter(result)
    return result


def select_vertices_edges(counter) -> Tuple[Edges, List[Any]]:
    def is_not_funct(word):
        if len(word) > 2:
            return True

        return word not in ",.)(:'\"`-?''``,%"

    #print("total pairs", len(counter))
    vertice_counter = get_vertices_info(counter)
    #print("total terms", len(vertice_counter))
    common_vertices = list([(k, cnt) for k, cnt in vertice_counter.items() if cnt > 100])
    common_vertices.sort(key=lambda x: x[1], reverse=True)
    # print(left(common_vertices[:20]))
    # print("Terms with more than 100 appearance : ", len(common_vertices))
    valid_vertices: List[Any] = lfilter(is_not_funct, left(common_vertices))
    valid_pairs = list([((a, b), cnt) for (a, b), cnt in counter.items()
                        if a in valid_vertices and b in valid_vertices])
    # print("valid pairs", len(valid_pairs))
    unnormalized_edges: Dict[Any, Dict] = {}
    for (a, b), cnt in valid_pairs:
        if a not in unnormalized_edges:
            unnormalized_edges[a] = Counter()
        unnormalized_edges[a][b] += cnt

    edges = {}
    for vertex_a, raw_edges in unnormalized_edges.items():
        total = sum(raw_edges.values())
        local_edges = Counter()
        for vertex_b, cnt in raw_edges.items():
            prob = cnt / total
            local_edges[vertex_b] = prob
        edges[vertex_a] = local_edges
    return Edges(edges), valid_vertices


class RandomWalkWorker(job_runner.WorkerInterface):
    def __init__(self, out_path, input_name):
        self.out_dir = out_path
        self.pc_co_occurrence: List[Tuple[int, Counter]] = load_from_pickle(input_name)
        self.p_reset = 0.1
        self.max_repeat = 1000

    def work(self, job_id):
        cid, pair_counter = self.pc_co_occurrence[job_id]
        result: Counter = run_random_walk_wrap(self.max_repeat, self.p_reset, pair_counter)
        output = cid, result
        save_path = os.path.join(self.out_dir, str(job_id))
        pickle.dump(output, open(save_path, "wb"))

