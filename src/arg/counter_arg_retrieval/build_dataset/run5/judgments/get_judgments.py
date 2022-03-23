from typing import Dict, List

from arg.counter_arg_retrieval.build_dataset.data_prep.remove_duplicate_passages import SplitDocDict
from arg.counter_arg_retrieval.build_dataset.judgement_common import convert_to_judgment_ex_entries
from arg.counter_arg_retrieval.build_dataset.judgments import JudgementEx
from arg.counter_arg_retrieval.build_dataset.run5.passage_scoring_util import load_run5_swtt_passage_as_d
from arg.counter_arg_retrieval.build_dataset.run5.path_helper import get_ranked_list_path_to_annotate, load_qids
from misc_lib import tprint
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def load_ranked_list_to_annotate(run_name) -> Dict[str, List[TrecRankedListEntry]]:
    save_path = get_ranked_list_path_to_annotate(run_name)
    try:
        ret = load_ranked_list_grouped(save_path)
        return ret
    except ValueError:
        print("Error while parsing: {}".format(save_path))
        raise


def get_judgments_todo() -> List[JudgementEx]:
    runs = ["PQ_10", "PQ_11", "PQ_12", "PQ_13"]
    pq_for_run = {}
    for run_name in runs:
        tprint(run_name)
        pq = load_ranked_list_to_annotate(run_name)
        pq_for_run[run_name] = pq

    qids = load_qids()
    judgments_todo: List[JudgementEx] = []
    for qid in qids:
        tprint("Qid: " + qid)
        passages: SplitDocDict = load_run5_swtt_passage_as_d(qid)
        tprint("Loading swtt passages done")
        for run_name in runs:
            entries = pq_for_run[run_name][qid]
            per_query = {qid: entries}
            required_judgements: List[JudgementEx] = convert_to_judgment_ex_entries(passages, per_query)
            for e in required_judgements:
                if e not in judgments_todo:
                    judgments_todo.append(e)
    tprint("Total of {} judgments".format(len(judgments_todo)))
    return judgments_todo


def main():
    todo = get_judgments_todo()
    print("{} judgments to do ".format(len(todo)))


if __name__ == "__main__":
    main()