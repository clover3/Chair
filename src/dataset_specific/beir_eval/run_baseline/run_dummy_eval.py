from adhoc.retriever_if import NullRetriever
from dataset_specific.beir_eval.beir_common import beir_dataset_list_A
from dataset_specific.beir_eval.run_helper import run_retrieval_and_eval_on_beir


for dataset in beir_dataset_list_A:
    print("Running for %s", dataset)
    retriever = NullRetriever()
    max_doc_per_list = 1000
    method = "null"
    split = "test"
    run_retrieval_and_eval_on_beir(dataset, split, method, retriever, max_doc_per_list)