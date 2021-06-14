from arg.bm25 import BM25
from cache import load_from_pickle
from tab_print import print_table
from tlm.data_gen.msmarco_doc_gen.max_sent_encode import SegScorer, PassageScoreTuner
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10docMulti


if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource10docMulti(split)
    max_seq_length = 512
    job_id = 0
    df = load_from_pickle("mmd_df_10")
    avdl_raw = 1350
    avdl_passage = 40

    rows = []
    k1 = 0.1
    for avdl in [10, 40, 100, 200]:
        bm25 = BM25(df, avdl=avdl, num_doc=321384, k1=k1, k2=100, b=0.75)
        scorer = SegScorer(bm25, max_seq_length)
        qids = resource.query_group[job_id]
        tuner = PassageScoreTuner(resource, scorer)
        row = [avdl, tuner.get_mrr(qids)]
        rows.append(row)

    print_table(rows)
