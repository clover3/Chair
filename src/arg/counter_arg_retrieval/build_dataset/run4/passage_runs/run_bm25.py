from arg.counter_arg_retrieval.build_dataset.job_running import run_job_runner_json
from arg.counter_arg_retrieval.build_dataset.methods.bm25_clue import build_bm25
from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI, FutureScorerTokenBased
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.bm25_interface import BM25Scorer
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.run3_util import load_premise_queries
from arg.counter_arg_retrieval.build_dataset.run3.swtt.save_trec_style import read_pickled_predictions_and_save
from arg.counter_arg_retrieval.build_dataset.run4.run4_util import Run4PassageScoring


def save_to_ranked_list():
    run_name = "PQ_9"
    read_pickled_predictions_and_save(run_name)


def main():
    query_list = load_premise_queries()
    bm25 = BM25Scorer(build_bm25())
    scorer: FutureScorerI = FutureScorerTokenBased(bm25)
    scoring = Run4PassageScoring(scorer)
    run_job_runner_json(query_list, scoring.work, "PQ_9")


if __name__ == "__main__":
    main()