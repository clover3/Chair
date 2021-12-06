import os
from typing import List, Dict, Tuple
from typing import NamedTuple

from arg.counter_arg_retrieval.build_dataset.ca_query import CAQuery
from arg.counter_arg_retrieval.build_dataset.run3.misc_qid import CA3QueryIDGen
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from cache import load_pickle_from
from cpath import output_path
from list_lib import left, idx_by_fn
from misc_lib import get_dir_files
from tab_print import save_table_as_csv
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def tuple_compare(tuple1, tuple2):
    num_dim = 5
    for j in range(num_dim):
        v1 = tuple1[j]
        v2 = tuple2[j]
        if v1 != v2:
            if v1 < v2:
                return -1
            else:
                return +1
    return 0


class JudgeEntry(NamedTuple):
    p_text: str
    c_text: str
    doc_id: str
    passage_idx: int
    passage_html: str


class PassageID(NamedTuple):
    doc_id: str
    passage_idx: int


def read_save(prediction_entries: List[Tuple[CAQuery, List[Tuple[str, SWTTScorerOutput]]]],
              ranked_list_d: Dict[str, List[TrecRankedListEntry]],
              csv_save_path,
              ):
    query_id_gen = CA3QueryIDGen()

    head = ['qid', 'p_text', 'c_text', 'doc_id', 'passage_idx', 'passage']
    query_dict = idx_by_fn(query_id_gen.get_qid, left(prediction_entries))
    scored_doc_dicts = {}
    for query, items in prediction_entries:
        scored_doc_dicts[query_id_gen.get_qid(query)] = items

    rows = []
    for query_id, ranked_list in ranked_list_d.items():
        query = query_dict[query_id]
        p_text = query.perspective
        c_text = query.claim
        scored_docs_d = dict(scored_doc_dicts[query_id])
        for e in ranked_list:
            doc_id, passage_idx_s = e.doc_id.split("_")
            passage_idx = int(passage_idx_s)
            word_tokens_list = scored_docs_d[doc_id].get_passage(passage_idx)
            passage_html = get_word_tokens_to_html(word_tokens_list)

            row = [query_id, p_text, c_text, doc_id, passage_idx_s, passage_html]
            rows.append(row)
    save_table_as_csv([head] + rows, csv_save_path)


def get_candidate_passages(docs_and_scores, duplicate_doc_ids) -> List[Tuple[PassageID, float]]:
    judge_candidates = []
    for doc_id, scores in docs_and_scores:
        if doc_id in duplicate_doc_ids:
            continue
        for passage_idx, score in enumerate(scores.scores):
            e = PassageID(doc_id, passage_idx), score
            judge_candidates.append(e)
    return judge_candidates


#
# st, ed = scores.windows_st_ed_list[passage_idx]
# word_tokens_grouped: List[List[str]] = doc.get_word_tokens_grouped(st, ed)
def get_word_tokens_to_html(word_tokens_grouped):
    text_html_all = ""
    for word_tokens in word_tokens_grouped:
        text_html = "<p>{}</p>".format(" ".join(word_tokens))
        text_html_all += text_html
    return text_html_all


def read_save_default(run_name):
    csv_save_path = os.path.join(output_path, "ca_building", "run3", "csv", "{}.csv".format(run_name))
    prediction_entries = load_entries_from_run3_dir(run_name)
    sliced_ranked_list_path = os.path.join(output_path, "ca_building", "run3", "passage_ranked_list_sliced",
                                           "{}.txt".format(run_name))
    ranked_list_d = load_ranked_list_grouped(sliced_ranked_list_path)
    read_save(prediction_entries, ranked_list_d, csv_save_path)


def load_entries_from_run3_dir(run_name):
    prediction_entries: List[Tuple[CAQuery, List[Tuple[str, SWTTScorerOutput]]]] = []
    save_dir = os.path.join(output_path, "ca_building", "run3", run_name)
    for file_path in get_dir_files(save_dir):
        prediction_entries.extend(load_pickle_from(file_path))
    return prediction_entries


def main():
    read_save_default("PQ_1")


if __name__ == "__main__":
    main()

