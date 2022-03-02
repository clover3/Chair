from typing import List

from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from arg.counter_arg_retrieval.build_dataset.run5.passage_scoring_util import run_job_runner
from arg.counter_arg_retrieval.build_dataset.run5.path_helper import load_premise_queries
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText


class CountScorerTodo(FutureScorerI):
    def __init__(self):
        self.count = 0

    def get_score_future(self, query_text: str,
                         doc: SegmentwiseTokenizedText,
                         passages: List[PassageRange]):
        self.count += len(passages)

    def do_duty(self):
        print("{} items".format(self.count))


def main():
    query_list = load_premise_queries()
    scorer: FutureScorerI = CountScorerTodo()
    run_job_runner(query_list, scorer, "PQ_Dummy")


if __name__ == "__main__":
    main()
