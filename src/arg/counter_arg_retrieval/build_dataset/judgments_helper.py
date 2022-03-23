import random
from typing import List, Dict, Tuple

from arg.counter_arg_retrieval.build_dataset.annotation_prep import get_word_tokens_to_html
from arg.counter_arg_retrieval.build_dataset.ca_query import CATask
from arg.counter_arg_retrieval.build_dataset.judgments import Judgement, JudgementEx
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from list_lib import index_by_fn
from tab_print import save_table_as_csv
from trec.types import DocID


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


def save_judgement_entries_from_ex(
        judgement_entries: List[JudgementEx],
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
        swtt: SegmentwiseTokenizedText = e.swtt
        st = e.passage_st
        ed = e.passage_ed
        passage_idx_s = str(e.passage_idx)
        st_s = str(e.passage_st)
        ed_s = str(e.passage_ed)
        word_tokens_list = swtt.get_word_tokens_grouped(st, ed)
        passage_html = get_word_tokens_to_html(word_tokens_list)
        row = [query_id, p_text, query.p_stance,
               c_text, query.c_stance, query.entity,
               query.pc_stance,
               e.doc_id, st_s, ed_s, passage_idx_s, passage_html]
        rows.append(row)
    random.shuffle(rows)
    save_table_as_csv([head] + rows, csv_save_path)