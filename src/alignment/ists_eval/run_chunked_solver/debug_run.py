import os
import sys
from typing import List

from alignment.ists_eval.chunked_solver.nli_partial import PartialNLIDrivenSolver
from cpath import data_path
from dataset_specific.ists.parse import iSTSProblemWChunk
from dataset_specific.ists.path_helper import load_ists_problems_w_chunk
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, \
    get_pep_cache_client

word2vec_path = os.path.join(data_path, "GoogleNews-vectors-negative300.bin")


def pep():
    problem_id_target = sys.argv[1]
    genre = "headlines"
    split = "train"
    nli_predictor: NLIPredictorSig = get_pep_cache_client()
    solver = PartialNLIDrivenSolver(nli_predictor, word2vec_path, verbose=True)
    problems: List[iSTSProblemWChunk] = load_ists_problems_w_chunk(genre, split)
    target_problem = [p for p in problems if p.problem_id == problem_id_target]
    solver.batch_solve(target_problem)


if __name__ == "__main__":
    pep()
