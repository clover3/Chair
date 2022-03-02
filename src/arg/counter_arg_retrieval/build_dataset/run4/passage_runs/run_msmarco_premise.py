from arg.counter_arg_retrieval.build_dataset.job_running import run_job_runner
from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.passage_scoring.get_scorers import get_msmarco_future_scorer
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.run3_util import load_premise_queries
from arg.counter_arg_retrieval.build_dataset.run4.run4_util import Run4PassageScoring


def main():
    scorer: FutureScorerI = get_msmarco_future_scorer()
    query_list = load_premise_queries()
    scoring = Run4PassageScoring(scorer)
    run_job_runner(query_list, scoring.work, "PQ_6")


if __name__ == "__main__":
    main()
