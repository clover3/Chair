import logging

from adhoc.bm25_class import BM25
from trainer_v2.per_project.tli.qa_scorer.bm25_system import build_stats, BM25TextPairScorerClueWeb
from trainer_v2.per_project.tli.bioclaim_qa.eval_helper import batch_solve_bioclaim, \
    get_bioclaim_retrieval_corpus
from trainer_v2.per_project.tli.qa_scorer.nli_token_system import NLIBasedRelevance, NLIBasedRelevanceMultiSeg
from trainer_v2.per_project.tli.bioclaim_qa.path_helper import get_retrieval_save_path
from list_lib import right
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, get_pep_client, get_cached_client, \
    get_nli14_cache_client
from trec.trec_parse import write_trec_ranked_list_entry
from cpath import output_path
from misc_lib import path_join


g_pep_client = None
def get_pep_cache_client(hooking_fn=None) -> NLIPredictorSig:
    global g_pep_client
    if g_pep_client is None:
        forward_fn_raw: NLIPredictorSig = get_pep_client()
        sqlite_path = path_join(output_path, "nli", "bioclaim_nlits")
        cache_client = get_cached_client(forward_fn_raw, hooking_fn, sqlite_path)
        g_pep_client = cache_client
    else:
        cache_client = g_pep_client
    return cache_client.predict


def solve_save_bioclaim_w_nli(nli_predict_fn, run_name, split, idf_weighting=True):
    _, claims = get_bioclaim_retrieval_corpus(split)
    df, cdf, avdl = build_stats(right(claims))
    if idf_weighting:
        bm25 = BM25(df, avdl=avdl, num_doc=cdf, k1=0.00001, k2=100, b=0.5,
                    drop_stopwords=True)
        module = NLIBasedRelevance(nli_predict_fn, bm25)
    else:
        module = NLIBasedRelevance(nli_predict_fn)
    rl_flat = batch_solve_bioclaim(module.batch_predict, split, run_name)
    save_name = f"{run_name}_{split}"
    write_trec_ranked_list_entry(rl_flat, get_retrieval_save_path(save_name))


def solve_save_bioclaim_w_nli_enum(nli_predict_fn, run_name, split, idf_weighting=True):
    _, claims = get_bioclaim_retrieval_corpus(split)
    df, cdf, avdl = build_stats(right(claims))
    if idf_weighting:
        bm25 = BM25(df, avdl=avdl, num_doc=cdf, k1=0.00001, k2=100, b=0.5,
                    drop_stopwords=True)
        module = NLIBasedRelevanceMultiSeg(nli_predict_fn, bm25)
    else:
        module = NLIBasedRelevanceMultiSeg(nli_predict_fn)
    rl_flat = batch_solve_bioclaim(module.batch_predict, split, run_name)
    save_name = f"{run_name}_{split}"
    write_trec_ranked_list_entry(rl_flat, get_retrieval_save_path(save_name))


def solve_save_bioclaim_w_nli_clue_idf(nli_predict_fn, run_name, split):
    module = NLIBasedRelevance(nli_predict_fn, BM25TextPairScorerClueWeb().bm25)
    rl_flat = batch_solve_bioclaim(module.batch_predict, split, run_name)
    write_trec_ranked_list_entry(rl_flat, get_retrieval_save_path(run_name))


def common_run(split, nli_type, use_idf, do_enum):
    run_name = f"{nli_type}" + ("_idf" if use_idf else "")
    if do_enum:
        run_name = run_name + "_enum"

    print(run_name)
    if nli_type == "nli":
        nli_predict_fn = get_nli14_cache_client()
    elif nli_type == "nli_pep":
        nli_predict_fn = get_pep_cache_client()
    else:
        assert False

    if do_enum:
        solve_save_bioclaim_w_nli_enum(nli_predict_fn, run_name, split, use_idf)
    else:
        solve_save_bioclaim_w_nli(nli_predict_fn, run_name, split, use_idf)


def main():
    todo = [
        ("nli", False),
        ("nli", True),
        # ("nli_pep", False),
        # ("nli_pep", True)
    ]

    for nli_type in ["nli"]:
        for use_idf in [True, False]:
            for do_enum in [True, False]:
                common_run("dev", nli_type, use_idf, do_enum)
                common_run("test", nli_type, use_idf, do_enum)


if __name__ == "__main__":
    main()
