import os
from typing import NamedTuple, List, Tuple

from arg.counter_arg_retrieval.build_dataset.path_helper import load_sliced_passage_ranked_list
from arg.counter_arg_retrieval.build_dataset.run5.path_helper import load_premise_queries
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from trec.qrel_parse import load_qrels_structured


def join_ranked_list_w_qrel(qrels, run_name):
    pq = load_sliced_passage_ranked_list(run_name)
    for qid, entries in pq.items():
        try:
            qrel_per_qid = qrels[qid]
            for e in entries:
                try:
                    judged_score = qrel_per_qid[e.doc_id]
                    yield qid, e.doc_id, e.score, judged_score
                except KeyError:
                    pass
        except KeyError:
            pass


class _Entry(NamedTuple):
    qid: str
    q_tokens: List[str]
    passage_id: str
    model_score: float
    judged_score: int


def load_ca_run_data(run_name) -> List[_Entry]:
    # List[(qid, passage_id, passage_content, model score, judged score)]
    tokenizer = get_tokenizer()
    query_list: List[Tuple[str, str]] = load_premise_queries()
    query_d = dict(query_list)

    judgment_path = os.path.join(output_path, "ca_building", "qrel", "0522.txt")
    qrels = load_qrels_structured(judgment_path)
    for qid, passage_id, model_score, judged_score in join_ranked_list_w_qrel(qrels, run_name):
        query_text = query_d[qid]
        q_tokens = tokenizer.tokenize(query_text)
        e = _Entry(qid,
                   q_tokens,
                   passage_id,
                   model_score,
                   judged_score
               )
        yield e
