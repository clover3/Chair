import random
from typing import List, Tuple, Dict

from arg.counter_arg_retrieval.build_dataset.ca_query import CAQuery, CATask
from arg.counter_arg_retrieval.build_dataset.judgments import Judgement
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from arg.counter_arg_retrieval.build_dataset.run3.misc_qid import CA3QueryIDGen
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from list_lib import index_by_fn, left
from tab_print import save_table_as_csv
from trec.types import TrecRankedListEntry, DocID


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


def save_judgement_entries(
        judgement_entries: List[Judgement],
        passages: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]],
        ca_task: List[CATask],
        csv_save_path,
              ):
    ca_task_d: Dict[str, CATask] = index_by_fn(lambda c: c.qid, ca_task)

    head = ['qid', 'p_text', 'p_stance',
            'c_text', 'c_stance', 'entity', 'pc_stance', 'doc_id', 'passage_st', 'passage_ed',
            'passage_idx',
            'passage']

    rows = []
    for e in judgement_entries:
        query_id = e.qid
        query: CATask = ca_task_d[query_id]
        p_text: str = query.perspective
        c_text: str = query.claim
        doc, passage_ranges = passages[e.doc_id]
        st = e.passage_st
        ed = e.passage_ed
        passage_idx_s = str(e.passage_idx)
        st_s = str(e.passage_st)
        ed_s = str(e.passage_ed)
        word_tokens_list = doc.get_word_tokens_grouped(st, ed)
        passage_html = get_word_tokens_to_html(word_tokens_list)
        row = [query_id, p_text, query.p_stance,
               c_text, query.c_stance, query.entity,
               query.pc_stance,
               e.doc_id, st_s, ed_s, passage_idx_s, passage_html]
        rows.append(row)
    random.shuffle(rows)
    save_table_as_csv([head] + rows, csv_save_path)


def get_word_tokens_to_html(word_tokens_grouped):
    text_html_all = ""
    for word_tokens in word_tokens_grouped:
        text_html = "<p>{}</p>".format(" ".join(word_tokens))
        text_html_all += text_html
    return text_html_all