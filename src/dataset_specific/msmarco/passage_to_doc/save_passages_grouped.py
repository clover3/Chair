import os
from collections import defaultdict
from typing import List, Dict, Tuple

# pass
from cache import load_from_pickle, load_pickle_from
from cpath import data_path
from dataset_specific.msmarco.common import QueryID, load_query_group
from dataset_specific.msmarco.misc_tool import get_qid_to_job_id
from dataset_specific.msmarco.passage_common import enum_passage_corpus
from misc_lib import tprint, TimeEstimator, group_by, get_first


def main():
    PassageID = str
    positive_passage_list: List[Tuple[QueryID, PassageID, PassageID]] = load_from_pickle("msmarco_doc_joined_passage_triples")
    passage_to_qid: Dict[PassageID, List[QueryID]] = {}
    print("start")
    for qid, pid1, pid2 in positive_passage_list:
        pid1 = pid1.strip()
        pid2 = pid2.strip()
        if qid == "1000041":
            print(qid, pid1, pid2)
        if pid1 not in passage_to_qid:
            passage_to_qid[pid1] = list()
        passage_to_qid[pid1].append(qid)
        if pid2 not in passage_to_qid:
            passage_to_qid[pid2] = list()
        passage_to_qid[pid2].append(qid)

    tprint("{} passages to find".format(len(passage_to_qid)))
    query_group: List[List[QueryID]] = load_query_group("train")
    qid_to_job_id = get_qid_to_job_id(query_group)

    tprint("Enumerating corpus")
    job_grouped: Dict[int, List[Tuple[QueryID, PassageID, str]]] = defaultdict(list)
    n_record = 8841823
    n_found = 0
    ticker = TimeEstimator(n_record, "enum", 1000)
    for passage_id, content in enum_passage_corpus():
        ticker.tick()
        if passage_id in passage_to_qid:
            qids = passage_to_qid[passage_id]
            n_found += 1
            assert qids
            for qid in qids:
                job_id = qid_to_job_id[qid]
                e = qid, passage_id, content
                job_grouped[job_id].append(e)
        else:
            pass

    print("{} passages are actual found".format(n_found))
    # root_dir = os.path.join(data_path, "msmarco_passage_doc_grouped")
    # exist_or_mkdir(root_dir)
    # for job_id, qid_pid_content_list in job_grouped.items():
    #     save_path = os.path.join(root_dir, str(job_id))
    #     pickle.dump(qid_pid_content_list, open(save_path, "wb"))


def load_passage_d_for_job(job_id) -> Dict[QueryID, List[Tuple[str, str]]]:
    root_dir = os.path.join(data_path, "msmarco_passage_doc_grouped")
    save_path = os.path.join(root_dir, str(job_id))
    l: List[Tuple[QueryID, str, str]] = load_pickle_from(save_path)
    qid_grouped = group_by(l, get_first)
    output: Dict[QueryID, List[Tuple[str, str]]] = {}
    for qid, entries in qid_grouped.items():
        simple_l = list([(pid, content) for qid, pid, content in entries])
        output[qid] = simple_l
    print("for job {}, {} queries are loaded".format(job_id, len(output)))
    return output


if __name__ == "__main__":
    main()
