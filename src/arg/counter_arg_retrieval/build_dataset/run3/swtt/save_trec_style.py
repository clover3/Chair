import json
import os
from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.run3.run_interface.run3_util import json_to_scoring_output
from arg.counter_arg_retrieval.build_dataset.run3.swtt.print_to_csv import load_entries_from_run3_dir, \
    load_json_entries_from_run3_dir
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from cpath import output_path
from trec.ranked_list_util import assign_rank
from trec.trec_parse import write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def read_pickled_predictions_and_save(run_name):
    prediction_entries: List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]] = \
        load_entries_from_run3_dir(run_name)
    flat_entries = []
    for qid, docs_and_scores in prediction_entries:
        per_qid_ranked_list = []
        for doc_id, swtt_scorer_output in docs_and_scores:
            for idx, s in enumerate(swtt_scorer_output.scores):
                new_doc_id = "{}_{}".format(doc_id, idx)
                out_e = TrecRankedListEntry(qid, new_doc_id, 0, s, run_name)
                per_qid_ranked_list.append(out_e)
        ranked_list: List[TrecRankedListEntry] = assign_rank(per_qid_ranked_list)
        flat_entries.extend(ranked_list)
    save_path = os.path.join(output_path, "ca_building", "run3", "passage_ranked_list", "{}.txt".format(run_name))
    write_trec_ranked_list_entry(flat_entries, save_path)


def read_json_predictions_and_save(run_name):
    def parse_json(path):
        j = json.load(open(path, "r"))
        return json_to_scoring_output(j)
    prediction_entries: List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]] = \
        load_json_entries_from_run3_dir(run_name, parse_json)

    print("{} prediction_entries".format(len(prediction_entries)))
    flat_entries = []
    for qid, docs_and_scores in prediction_entries:
        per_qid_ranked_list = []
        for doc_id, swtt_scorer_output in docs_and_scores:
            for idx, s in enumerate(swtt_scorer_output.scores):
                new_doc_id = "{}_{}".format(doc_id, idx)
                out_e = TrecRankedListEntry(qid, new_doc_id, 0, s, run_name)
                per_qid_ranked_list.append(out_e)
        ranked_list: List[TrecRankedListEntry] = assign_rank(per_qid_ranked_list)
        flat_entries.extend(ranked_list)
    save_path = os.path.join(output_path, "ca_building", "run3", "passage_ranked_list", "{}.txt".format(run_name))
    write_trec_ranked_list_entry(flat_entries, save_path)


def main():
    # run_name = "PQ_1"
    # read_pickled_predictions_and_save(run_name)
    read_json_predictions_and_save("PQ_4")
    # read_pickled_predictions_and_save("PQ_3")


if __name__ == "__main__":
    main()
