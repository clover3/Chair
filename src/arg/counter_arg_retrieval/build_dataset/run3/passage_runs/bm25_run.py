import json
import os
import pickle

from arg.counter_arg_retrieval.build_dataset.job_running import run_job_runner_json
from arg.counter_arg_retrieval.build_dataset.methods.bm25_clue import build_bm25
from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI, FutureScorerTokenBased, \
    scoring_output_to_json
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.bm25_interface import BM25Scorer
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.run3_util import Run3PassageScoring, \
    load_premise_queries
from arg.counter_arg_retrieval.build_dataset.run3.swtt.save_trec_style import read_pickled_predictions_and_save
from cpath import output_path, cache_path


def save_to_ranked_list():
    run_name = "PQ_4"
    read_pickled_predictions_and_save(run_name)


def main():
    query_list = load_premise_queries()
    bm25 = BM25Scorer(build_bm25())
    scorer: FutureScorerI = FutureScorerTokenBased(bm25)
    scoring = Run3PassageScoring(scorer)
    run_job_runner_json(query_list, scoring.work, "PQ_4")


def debug():
    query_list = load_premise_queries()
    bm25 = BM25Scorer(build_bm25())
    scorer: FutureScorerI = FutureScorerTokenBased(bm25)
    scoring = Run3PassageScoring(scorer)
    obj = scoring.work(query_list[:1])
    obj_json = scoring_output_to_json(obj)
    json_save_path = os.path.join(output_path, "debug.json")
    json.dump(obj_json, open(json_save_path, "w"), indent=True)


def debug2():
    # json_save_path = os.path.join(output_path, "debug.json")
    # j = json.load(open(json_save_path, "r"))
    # output = json_to_scoring_output(j)
    # load_from_pickle("bm25_run_Debug_obj")
    pickle_name = "{}.pickle".format("bm25_run_Debug_obj")
    path = os.path.join(cache_path, pickle_name)
    obj = pickle.load(open(path, "rb"), errors="ignore")


if __name__ == "__main__":
    main()
