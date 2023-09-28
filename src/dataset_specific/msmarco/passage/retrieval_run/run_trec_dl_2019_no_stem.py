from dataset_specific.msmarco.passage.doc_indexing.retriever import get_mmp_bm25_retriever, \
    get_mmp_bm25_retriever_no_stem
from dataset_specific.msmarco.passage.path_helper import load_mmp_test_queries, TREC_DL_2019
from dataset_specific.msmarco.passage.retrieval_run.run_trec_dl_2019 import run_mmp_test_retrieval


def main():
    # Run BM25 retrieval
    dataset = TREC_DL_2019
    method = "BM25_no_stem"
    retriever = get_mmp_bm25_retriever_no_stem()
    run_mmp_test_retrieval(dataset, method, retriever)



if __name__ == "__main__":
    main()