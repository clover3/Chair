import os
import xmlrpc.client
from typing import List, Iterable, Callable, Dict, Tuple, Set

from cpath import output_path
from misc_lib import path_join, get_dir_files
from trec.ranked_list_util import build_ranked_list
from trec.trec_parse import write_trec_ranked_list_entry


def read_passages() -> List[Tuple[str, str, List[Tuple[str, str]]]]:
    rerank_dir = path_join(output_path, "transparency", "msmarco", "rerank_jobs")
    passage_payload = []
    for file_path in get_dir_files(rerank_dir):
        qid = os.path.basename(file_path)
        lines = open(file_path, "r").readlines()
        lines = [l for l in lines if l.strip()]
        query_text = lines[0]
        cursor = 1
        passages = []
        while cursor+2 < len(lines):
            bar = lines[cursor]
            doc_id = int(lines[cursor+1])
            text = lines[cursor+2]
            passages.append((str(doc_id), text))
            cursor += 3
        passage_payload.append((qid, query_text, passages))

    return passage_payload


def main():
    port = 28122
    proxy = xmlrpc.client.ServerProxy('http://localhost:{}'.format(port))

    passages_todo = read_passages()
    run_name = "splade"

    payload = []
    for qid, query_text, passage in passages_todo:
        for doc_id, text in passage:
            payload.append((query_text, text))

    scores: List[float] = proxy.predict(payload)
    assert len(scores) == len(payload)
    ranked_list = []
    score_d = dict(zip(payload, scores))
    for qid, query_text, passage in passages_todo:
        scored_docs: List[Tuple[str, float]] = []
        for doc_id, text in passage:
            key = query_text, text
            score = score_d[key]
            scored_docs.append((doc_id, score))

        ranked_list.extend(build_ranked_list(qid, run_name, scored_docs))

    save_path = path_join(output_path, "transparency", "msmarco", run_name + ".txt")
    write_trec_ranked_list_entry(ranked_list, save_path)


if __name__ == "__main__":
    main()