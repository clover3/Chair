import os
from typing import List, Dict

from arg.counter_arg_retrieval.build_dataset.run5.path_helper import load_swtt_jsonl_per_job, \
    get_swtt_per_query_path, load_raw_rlg, load_str_swtt_pair_list
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cache import save_list_to_jsonl, StrItem
from cpath import output_path
from misc_lib import tprint, exist_or_mkdir


def get_swtt_per_query_temp_path(query_id, job_no):
    save_dir = os.path.join(output_path, "ca_building", "run5", "per_query_doc_temp_swtt")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, "{}_{}.jsonl".format(query_id, job_no))
    return save_path


def main():
    rlg = load_raw_rlg()

    num_jobs = 34
    for job_no in range(num_jobs):
        tprint("Loading for job {}".format(job_no))
        obj: Dict[str, SegmentwiseTokenizedText] = load_swtt_jsonl_per_job(job_no)
        tprint("Loaded {} docs".format(len(obj)))
        docs = obj

        for query_id, ranked_list in rlg.items():
            docs_per_query: List[StrItem] = []
            miss = 0
            for e in ranked_list:
                try:
                    doc: SegmentwiseTokenizedText = docs[e.doc_id]
                    docs_per_query.append(StrItem(e.doc_id, doc))
                except KeyError:
                    miss += 1
                    pass
            tprint("Miss rate {0:.2f}".format(miss / len(ranked_list)))
            save_path = get_swtt_per_query_temp_path(query_id, job_no)
            save_list_to_jsonl(docs_per_query, save_path)


def merge_per_query():
    rlg = load_raw_rlg()
    num_jobs = 34
    for query_id, ranked_list in rlg.items():

        tprint("Loading for query_id {}".format(query_id))
        per_query_items: List[StrItem[SegmentwiseTokenizedText]] = []
        for job_no in range(num_jobs):
            save_path = get_swtt_per_query_temp_path(query_id, job_no)
            items: List[StrItem[SegmentwiseTokenizedText]] = load_str_swtt_pair_list(save_path)
            per_query_items.extend(items)

        save_list_to_jsonl(per_query_items, get_swtt_per_query_path(query_id))


if __name__ == "__main__":
    merge_per_query()
