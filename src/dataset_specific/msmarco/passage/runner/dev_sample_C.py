from cpath import output_path
from dataset_specific.msmarco.passage.runner.dev_sample1000 import sample_query_doc
from misc_lib import path_join
from table_lib import tsv_iter


def get_exclude_qids() -> list[str]:
    dev1000_query_path = path_join("data", "msmarco", "dev_sample1000", "queries.tsv")
    dev1000_qids = [qid for qid, query in tsv_iter(dev1000_query_path)]
    return dev1000_qids


def main():
    subset_name = "sample_dev_C"
    source_corpus_path = path_join("data", "msmarco", "top1000.dev")
    sample_corpus_save_path = path_join("data", "msmarco", subset_name, "corpus.tsv")
    sample_query_save_path = path_join("data", "msmarco", subset_name, "queries.tsv")
    top1000_iter = tsv_iter(source_corpus_path)
    exclude_qids = set(get_exclude_qids())

    def is_exclude(e):
        qid, _pid, _q_text, _p_text = e
        return qid not in exclude_qids

    itr = filter(is_exclude, top1000_iter)

    n_query = 100
    print(f"subset_name: {subset_name} n_query={n_query}")
    sample_query_doc(itr, n_query,
                     sample_corpus_save_path, sample_query_save_path)


if __name__ == "__main__":
    main()


