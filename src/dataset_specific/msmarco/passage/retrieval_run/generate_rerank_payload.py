from adhoc.json_run_eval_helper import load_json_qres
from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_collection
from dataset_specific.msmarco.passage.path_helper import load_mmp_test_queries, TREC_DL_2019
from typing import List, Iterable, Callable, Dict, Tuple, Set


def generate_rerank_payload(run_name, dataset):
    qres: Dict[str, Dict[str, float]] = load_json_qres(run_name)
    queries_d: Dict[str, str] = dict(load_mmp_test_queries(dataset))
    save_path = get_rerank_payload_save_path(run_name, dataset)

    all_doc_ids = set()
    for qid, entries in qres.items():
        all_doc_ids.update(entries.keys())

    doc_id_to_text = {}
    for doc_id, text in load_msmarco_collection():
        if doc_id in all_doc_ids:
            doc_id_to_text[doc_id] = text

    with open(save_path, "w") as f:
        for qid, entries in qres.items():
            query_text = queries_d[qid]
            for doc_id, _score in entries.items():
                doc_text = doc_id_to_text[doc_id]
                row = qid, doc_id, query_text, doc_text
                f.write("\t".join(row)+ "\n")


def main():
    generate_rerank_payload("bm25", TREC_DL_2019)


if __name__ == "__main__":
    main()



