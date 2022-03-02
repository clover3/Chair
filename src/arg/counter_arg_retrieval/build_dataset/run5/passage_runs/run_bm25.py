from arg.counter_arg_retrieval.build_dataset.methods.bm25_clue import build_bm25
from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI, FutureScorerTokenBased
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.bm25_interface import BM25Scorer
from arg.counter_arg_retrieval.build_dataset.run5.passage_scoring_util import run_job_runner
from arg.counter_arg_retrieval.build_dataset.run5.path_helper import load_premise_queries


def main():
    query_list = load_premise_queries()
    bm25 = BM25Scorer(build_bm25())
    scorer: FutureScorerI = FutureScorerTokenBased(bm25)
    run_job_runner(query_list, scorer, "PQ_10")


if __name__ == "__main__":
    main()
