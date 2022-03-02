from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.ca_query import CAQuery
from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import load_json_file_convert_to_scoring_output
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from misc_lib import get_dir_files, tprint
from trec.ranked_list_util import assign_rank
from trec.trec_parse import write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def load_predictions_from_dir_path(parse_fn, save_dir):
    prediction_entries: List[Tuple[CAQuery, List[Tuple[str, SWTTScorerOutput]]]] = []
    for file_path in get_dir_files(save_dir):
        if file_path.endswith(".json"):
            prediction_entries.extend(parse_fn(file_path))
    return prediction_entries


def convert_prediction_entries_to_trec_entries(
        prediction_entries: List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]],
        run_name) -> List[TrecRankedListEntry]:
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
    return flat_entries


def convert_json_prediction_to_trec_style(json_file_path, run_name, save_path):
    tprint("convert_json_prediction_to_trec_style()")
    tprint("load_json_file_convert_to_scoring_output")
    entries: List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]]\
        = load_json_file_convert_to_scoring_output(json_file_path)
    tprint("convert_prediction_entries_to_trec_entries")
    t_entries: List[TrecRankedListEntry] = convert_prediction_entries_to_trec_entries(entries, run_name)
    write_trec_ranked_list_entry(t_entries, save_path)
