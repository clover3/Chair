import os

from arg.bm25 import BM25
from cache import load_from_pickle
from cpath import at_output_dir
from dataset_specific.msmarco.common import MSMarcoDataReader, at_working_dir, load_per_query_docs, MSMarcoDoc, \
    load_queries, QueryID
from misc_lib import get_second, TimeEstimator
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker, PointwiseGen
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource
from trec.trec_parse import load_ranked_list, load_ranked_list_grouped, write_trec_ranked_list_entry
from typing import List, Iterable, Callable, Dict, Tuple, Set

from trec.types import TrecRankedListEntry


def get_bm25_module():
    df = load_from_pickle("mmd_df_100")
    return BM25(df, avdl=1350, num_doc=321384, k1=1.2, k2=100, b=0.75)


def main():
    split = "dev"
    query_d = dict(load_queries(split))
    bm25_module = get_bm25_module()
    ranked_list_path = at_working_dir("msmarco-doc{}-top100".format(split))
    run_name = "BM25_df100"
    rlg = load_ranked_list_grouped(ranked_list_path)
    save_path = at_output_dir("ranked_list", "mmd_dev_{}.txt".format(run_name))
    te = TimeEstimator(100)
    out_entries = []
    for query_id, entries in rlg.items():
        doc_ids = list([e.doc_id for e in entries])
        docs = load_per_query_docs(query_id, None)

        found_doc_ids = list([d.doc_id for d in docs])
        not_found_doc_ids = list([doc_id for doc_id in doc_ids if doc_id not in found_doc_ids])
        doc_id_len = len(not_found_doc_ids)
        if doc_id_len:
            print("{} docs not found".format(doc_id_len))

        query_text = query_d[QueryID(query_id)]

        def score(doc: MSMarcoDoc):
            content = doc.title + " " + doc.body
            return bm25_module.score(query_text, content)

        scored_docs = list([(d, score(d)) for d in docs])
        scored_docs.sort(key=get_second, reverse=True)

        reranked_entries = []
        for rank, (doc, score) in enumerate(scored_docs):
            e = TrecRankedListEntry(query_id, doc.doc_id, rank, score, run_name)
            reranked_entries.append(e)
        out_entries.extend(reranked_entries)
        te.tick()

        if len(out_entries) > 100 * 100:
            break

    write_trec_ranked_list_entry(out_entries, save_path)


if __name__ == "__main__":
    main()
