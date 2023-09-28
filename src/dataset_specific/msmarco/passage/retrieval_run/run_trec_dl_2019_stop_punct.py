from adhoc.json_run_eval_helper import run_retrieval, save_json_qres
from dataset_specific.msmarco.passage.doc_indexing.retriever import get_mmp_bm25_retriever, \
    get_mmp_bm25_retriever_stemmed_stop_puntc
from dataset_specific.msmarco.passage.path_helper import load_mmp_test_queries, TREC_DL_2019


def run_mmp_test_retrieval(dataset, method, retriever):
    run_name = f"{dataset}_{method}"
    queries = load_mmp_test_queries(dataset)
    max_doc_per_query = 1000
    doc_score_d = run_retrieval(retriever, queries, max_doc_per_query)
    save_json_qres(run_name, doc_score_d)


def main():
    # Run BM25 retrieval
    dataset = TREC_DL_2019
    method = "BM25_stem_stop_punct"
    retriever = get_mmp_bm25_retriever_stemmed_stop_puntc()
    run_mmp_test_retrieval(dataset, method, retriever)



if __name__ == "__main__":
    main()