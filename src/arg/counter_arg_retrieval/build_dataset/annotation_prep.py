from typing import List, Tuple, Dict

from arg.counter_arg_retrieval.build_dataset.ca_query import CAQuery
from arg.counter_arg_retrieval.build_dataset.run3.misc_qid import CA3QueryIDGen
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from list_lib import index_by_fn, left
from tab_print import save_table_as_csv
from trec.types import TrecRankedListEntry


def read_save(prediction_entries: List[Tuple[CAQuery, List[Tuple[str, SWTTScorerOutput]]]],
              ranked_list_d: Dict[str, List[TrecRankedListEntry]],
              csv_save_path,
              ):
    query_id_gen = CA3QueryIDGen()

    head = ['qid', 'p_text', 'c_text', 'doc_id', 'passage_idx', 'passage']
    query_dict: Dict[str, CAQuery] = index_by_fn(query_id_gen.get_qid, left(prediction_entries))
    scored_doc_dicts = {}
    for query, items in prediction_entries:
        scored_doc_dicts[query_id_gen.get_qid(query)] = items

    rows = []
    for query_id, ranked_list in ranked_list_d.items():
        query: CAQuery = query_dict[query_id]
        p_text: str = query.perspective
        c_text: str = query.claim
        scored_docs_d: Dict[str, SWTTScorerOutput] = dict(scored_doc_dicts[query_id])
        for e in ranked_list:
            doc_id, passage_idx_s = e.doc_id.split("_")
            passage_idx = int(passage_idx_s)
            word_tokens_list = scored_docs_d[doc_id].get_passage(passage_idx)
            passage_html = get_word_tokens_to_html(word_tokens_list)

            row = [query_id, p_text, c_text, doc_id, passage_idx_s, passage_html]
            rows.append(row)
    save_table_as_csv([head] + rows, csv_save_path)


def get_word_tokens_to_html(word_tokens_grouped):
    text_html_all = ""
    for word_tokens in word_tokens_grouped:
        text_html = "<p>{}</p>".format(" ".join(word_tokens))
        text_html_all += text_html
    return text_html_all