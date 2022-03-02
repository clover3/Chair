from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.passage_scoring.get_scorers import get_msmarco_future_scorer
from arg.counter_arg_retrieval.build_dataset.run5.passage_scoring_util import run_job_runner
from arg.counter_arg_retrieval.build_dataset.run5.path_helper import load_premise_queries


def main():
    query_list = load_premise_queries()
    scorer: FutureScorerI = get_msmarco_future_scorer()
    run_job_runner(query_list, scorer, "PQ_11")


if __name__ == "__main__":
    main()
