from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.run5.path_helper import load_swtt_jsonl_per_query, get_swtt_passage_path, \
    load_qids
from arg.counter_arg_retrieval.build_dataset.split_document_common import SplittedDoc, sd_to_json
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText, SWTTIndex
from bert_api.swtt.window_enum_policy import get_enum_policy_30_to_400_50per_doc
from cache import StrItem, save_list_to_jsonl_w_fn
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerF


def work_per_job(query_id):
    swtt_list: List[StrItem[SegmentwiseTokenizedText]] = load_swtt_jsonl_per_query(query_id)
    enum_policy = get_enum_policy_30_to_400_50per_doc()

    def convert(pair: Tuple[str, SegmentwiseTokenizedText], enum_policy):
        doc_id, doc = pair
        window_list: List[Tuple[SWTTIndex, SWTTIndex]] = enum_policy.window_enum(doc)
        return doc_id, doc, window_list

    result: List[SplittedDoc]\
        = [convert(e.to_tuple(), enum_policy) for e in swtt_list]

    save_path = get_swtt_passage_path(query_id)
    save_list_to_jsonl_w_fn(result, save_path, sd_to_json)


def main():
    qids = load_qids()
    num_jobs = len(qids)
    def work_per_job_wrap(job_id):
        return work_per_job(qids[job_id])

    runner = JobRunnerF(job_man_dir, num_jobs, "ca_building_run5_split_documents", work_per_job_wrap)
    runner.start()


if __name__ == "__main__":
    main()