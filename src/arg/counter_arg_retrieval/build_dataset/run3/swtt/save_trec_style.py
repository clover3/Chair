import os
from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.ca_query import CAQuery
from arg.counter_arg_retrieval.build_dataset.run3.misc_qid import CA3QueryIDGen
from arg.counter_arg_retrieval.build_dataset.run3.swtt.print_to_csv import load_entries_from_run3_dir
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from cpath import output_path
from trec.trec_parse import write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry, assign_rank


def main():
    run_name = "PQ_1"
    prediction_entries: List[Tuple[CAQuery, List[Tuple[str, SWTTScorerOutput]]]] = \
        load_entries_from_run3_dir(run_name)
    query_id_gen = CA3QueryIDGen()
    flat_entries = []
    for query, docs_and_scores in prediction_entries:
        qid = query_id_gen.get_qid(query)
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


if __name__ == "__main__":
    main()
