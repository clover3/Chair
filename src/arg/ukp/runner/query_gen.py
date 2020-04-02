from itertools import chain
from itertools import chain
from typing import List, Iterator

from arg.ukp.data_loader import load_all_data, UkpDataPoint
from arg.ukp.ukp_q_path import get_query_dir
from cache import save_to_pickle
from galagos.interface import DocQuery
from galagos.query_runs_ids import Q_CONFIG_ID_BM25_UKP
from galagos.tokenize_util import clean_tokenize_str_to_tokens
from list_lib import lmap
from misc_lib import exist_or_mkdir


def format_query(topic_tokens: List[str],
                 sent_tokens: List[str],
                 query_id: str,
                 topic_weight=2) -> DocQuery:
    k = 0
    weight = [topic_weight] * len(topic_tokens) + [1] * len(sent_tokens)
    weight_str = ":".join(["{}={}".format(idx, w) for idx, w in enumerate(weight)])
    all_tokens = topic_tokens + sent_tokens
    q_str_inner = " ".join(["#bm25:K={}({})".format(k, t) for t in all_tokens])
    query_str = "#combine:{}({})".format(weight_str, q_str_inner)
    return DocQuery({
        'number': query_id,
        'text': query_str
    })


def write_topic_sentence_as_query():
    query_collection_id = Q_CONFIG_ID_BM25_UKP

    dp_id_to_q_res_id = {}

    def dp_to_query(dp: UkpDataPoint) -> DocQuery:
        topic_tokens = clean_tokenize_str_to_tokens(dp.topic)
        sent_tokens = clean_tokenize_str_to_tokens(dp.sentence)
        qid = str(dp.id)
        dp_id_to_q_res_id[str(dp.id)] = "{}_{}".format(qid, query_collection_id)
        return format_query(topic_tokens, sent_tokens, qid, 3)

    train_data, val_data = load_all_data()

    def all_data_iterator() -> Iterator[UkpDataPoint]:
        for data_list in chain(train_data.values(), val_data.values()):
            for dp in data_list:
                yield dp

    all_queries: List[DocQuery] = lmap(dp_to_query, all_data_iterator())

    out_dir = get_query_dir(query_collection_id)
    exist_or_mkdir(out_dir)

    n_query_per_file = 50
    save_to_pickle(dp_id_to_q_res_id, "ukp_10_dp_id_to_q_res_id")
    #write_queries_to_files(n_query_per_file, out_dir, all_queries)

    # for each split,


if __name__ == "__main__":
    write_topic_sentence_as_query()

