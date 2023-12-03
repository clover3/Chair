from dataset_specific.msmarco.passage.doc_indexing.retriever import get_mmp_bm25_retriever_stemmed_stop_puntc
from dataset_specific.msmarco.passage.path_helper import TREC_DL_2019
from dataset_specific.msmarco.passage.trec_dl import run_mmp_retrieval


def main():
    # Run BM25 retrieval
    dataset = TREC_DL_2019
    method = "BM25_stem_stop_punct"
    retriever = get_mmp_bm25_retriever_stemmed_stop_puntc()
    run_mmp_retrieval(dataset, method, retriever)



if __name__ == "__main__":
    main()