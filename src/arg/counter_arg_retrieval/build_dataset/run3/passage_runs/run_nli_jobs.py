from arg.counter_arg_retrieval.build_dataset.job_running import run_job_runner
from arg.counter_arg_retrieval.build_dataset.passage_scoring.get_scorers import get_nli_scorer
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.run3_util import load_premise_queries, \
    Run3PassageScoring


def main():
    scorer = get_nli_scorer()
    query_list = load_premise_queries()
    scoring = Run3PassageScoring(scorer)
    run_job_runner(query_list, scoring.work, "PQ_3")


if __name__ == "__main__":
    main()
