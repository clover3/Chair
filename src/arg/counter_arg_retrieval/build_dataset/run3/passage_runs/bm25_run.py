from arg.bm25 import BM25
from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat_stemmed_from_pickle, cdf
from arg.counter_arg_retrieval.build_dataset.run3.passage_runs.run3_util import Run3PassageScoring, run_job_runner, \
    load_premise_queries
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.bm25_interface import BM25Scorer
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.passage_scorer import FutureScorerTokenBased, \
    FutureScorerI


def build_bm25():
    avdl = 10
    tf, df = load_clueweb12_B13_termstat_stemmed_from_pickle()
    return BM25(df, avdl=avdl, num_doc=cdf, k1=0.001, k2=100, b=0.5)


def main():
    query_list = load_premise_queries()
    bm25 = BM25Scorer(build_bm25())
    scorer: FutureScorerI = FutureScorerTokenBased(bm25)
    scoring = Run3PassageScoring(scorer)
    run_job_runner(query_list, scoring.work, "PQ_4")


if __name__ == "__main__":
    main()
