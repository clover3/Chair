import json
from typing import List, Dict, Tuple

from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI, rerank_passages, \
    unpack_futures, scoring_output_to_json
from arg.counter_arg_retrieval.build_dataset.run5.path_helper import get_ranked_list_per_query_path, load_qids, \
    get_passage_prediction_path, get_swtt_passage_path
from arg.counter_arg_retrieval.build_dataset.split_document_common import sd_from_json, SplittedDoc
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText, SWTTIndex
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from cache import load_list_from_jsonl
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerF
from trec.trec_parse import load_ranked_list_grouped


def load_run5_swtt_passage_as_d(query_id) -> Dict[str, Tuple[SegmentwiseTokenizedText, List[Tuple[SWTTIndex, SWTTIndex]]]]:
    save_path = get_swtt_passage_path(query_id)
    l: List[SplittedDoc] = load_list_from_jsonl(save_path, sd_from_json)

    out_d = {}
    for doc_id, doc, passage_indices in l:
        out_d[doc_id] = doc, passage_indices
    return out_d


def work_for_query(query_id: str, query_str: str,
                   scorer) -> List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]]:
    doc_as_passage_dict = load_run5_swtt_passage_as_d(query_id)
    rlg = load_ranked_list_grouped(get_ranked_list_per_query_path(query_id))
    query_list = [(query_id, query_str)]
    future_output = rerank_passages(doc_as_passage_dict, rlg, query_list, scorer)
    output = unpack_futures(future_output)
    return output


# swtt passage
# ranked list per_query
# qids
def run_job_runner(query_list: List[Tuple[str, str]],
                   scorer: FutureScorerI,
                   run_name: str):
    query_d = dict(query_list)
    qids = load_qids()

    def work_fn(job_id):
        qid = qids[job_id]
        save_path = get_passage_prediction_path(run_name, qid)
        query_text = query_d[qid]
        output = work_for_query(qid, query_text, scorer)
        j = scoring_output_to_json(output)
        json.dump(j, open(save_path, "w"), indent=True)

    job_name = "run5_{}".format(run_name)
    job_runner = JobRunnerF(job_man_dir, len(qids), job_name, work_fn)
    job_runner.start()


