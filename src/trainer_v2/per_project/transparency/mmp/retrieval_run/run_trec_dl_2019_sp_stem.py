from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import get_bm25_sp_stem_resource_path_helper
from dataset_specific.msmarco.passage.doc_indexing.retriever import get_bm25_retriever_from_conf
from dataset_specific.msmarco.passage.path_helper import TREC_DL_2019
from dataset_specific.msmarco.passage.trec_dl import run_mmp_test_retrieval_eval_report


def main():
    # Run BM25 retrieval
    dataset = TREC_DL_2019
    method = "BM25_sp_stem"
    conf = get_bm25_sp_stem_resource_path_helper()
    avdl = 52
    retriever = get_bm25_retriever_from_conf(conf, avdl)
    run_mmp_test_retrieval_eval_report(dataset, method, retriever)


if __name__ == "__main__":
    main()
