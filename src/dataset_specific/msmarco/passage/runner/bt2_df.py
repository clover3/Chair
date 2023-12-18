from typing import List, Iterable, Tuple

from adhoc.build_index import count_dl_df, save_df_dl
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import \
    get_bm25_bt2_resource_path_helper
from dataset_specific.msmarco.passage.runner.bt2_inv_index import iter_bert_tokenized_merged
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from typing import List, Iterable, Tuple

from adhoc.build_index import count_dl_df, save_df_dl
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import \
    get_bm25_bt2_resource_path_helper
from dataset_specific.msmarco.passage.runner.bt2_inv_index import iter_bert_tokenized_merged
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log


def common_index_preprocessing():
    conf = get_bm25_bt2_resource_path_helper()
    collection_size = 8841823
    itr = iter_bert_tokenized_merged()
    corpus_tokenized: Iterable[Tuple[str, List[str]]] = itr
    c_log.info("Counting df")
    outputs = count_dl_df(
        corpus_tokenized,
        num_docs=collection_size
    )
    c_log.info("Done")
    save_df_dl(conf, outputs)


def main():
    with JobContext("df_counting"):
        common_index_preprocessing()


if __name__ == "__main__":
    main()
