import json
import os
from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.passage_prediction_summarizer import \
    convert_prediction_entries_to_trec_entries
from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import json_to_scoring_output
from arg.counter_arg_retrieval.build_dataset.run3.swtt.print_to_csv import load_entries_from_jobs_dir, \
    load_json_entries_from_jobs_dir
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from cpath import output_path
from trec.trec_parse import write_trec_ranked_list_entry


def read_pickled_predictions_and_save(run_name):
    prediction_entries: List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]] = \
        load_entries_from_jobs_dir(run_name)
    flat_entries = convert_prediction_entries_to_trec_entries(prediction_entries, run_name)
    save_path = os.path.join(output_path, "ca_building", "passage_ranked_list", "{}.txt".format(run_name))
    write_trec_ranked_list_entry(flat_entries, save_path)


def read_json_predictions_and_save(run_name):
    save_path = os.path.join(output_path, "ca_building", "passage_ranked_list", "{}.txt".format(run_name))
    def parse_json(path):
        j = json.load(open(path, "r"))
        return json_to_scoring_output(j)
    prediction_entries: List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]] = \
        load_json_entries_from_jobs_dir(run_name, parse_json)

    print("{} prediction_entries".format(len(prediction_entries)))
    flat_entries = convert_prediction_entries_to_trec_entries(prediction_entries, run_name)
    write_trec_ranked_list_entry(flat_entries, save_path)


def main():
    # run_name = "PQ_1"
    # read_pickled_predictions_and_save(run_name)
    read_pickled_predictions_and_save("PQ_6")
    read_pickled_predictions_and_save("PQ_7")
    read_pickled_predictions_and_save("PQ_8")
    read_json_predictions_and_save("PQ_9")
    # read_pickled_predictions_and_save("PQ_3")


if __name__ == "__main__":
    main()
