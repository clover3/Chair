import logging

from adhoc.bm25_class import BM25
from trainer_v2.per_project.tli.bioclaim_qa.bm25_system import build_stats, BM25Clueweb
from trainer_v2.per_project.tli.bioclaim_qa.eval_helper import solve_bioclaim, batch_solve_bioclaim, \
    solve_bio_claim_and_save, get_bioclaim_retrieval_corpus
from trainer_v2.per_project.tli.bioclaim_qa.nli_token_system import NLIBasedRelevance, NLIBasedRelevanceMultiSeg
from trainer_v2.per_project.tli.bioclaim_qa.path_helper import get_retrieval_save_path
from list_lib import right
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, get_pep_client, get_cached_client, \
    get_nli14_cache_client
from trec.trec_parse import write_trec_ranked_list_entry
from cpath import output_path
from misc_lib import path_join


def exist_c_log():
    global c_lod



def get_pep_cache_client(hooking_fn=None) -> NLIPredictorSig:
    forward_fn_raw: NLIPredictorSig = get_pep_client()
    sqlite_path = path_join(output_path, "nli", "bioclaim_nlits")
    cache_client = get_cached_client(forward_fn_raw, hooking_fn, sqlite_path)
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
    write_trec_ranked_list_entry(rl_flat, get_retrieval_save_path(run_name))


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
    write_trec_ranked_list_entry(rl_flat, get_retrieval_save_path(run_name))


def solve_save_bioclaim_w_nli_clue_idf(nli_predict_fn, run_name, split):
    module = NLIBasedRelevance(nli_predict_fn, BM25Clueweb().bm25)
    rl_flat = batch_solve_bioclaim(module.batch_predict, split, run_name)
    write_trec_ranked_list_entry(rl_flat, get_retrieval_save_path(run_name))


def nli_dev():
    split = "dev"
    run_name = "pep_idf"
    nli_predict_fn = get_pep_cache_client()

    solve_save_bioclaim_w_nli(nli_predict_fn, run_name, split)


def nli14():
    split = "dev"
    run_name = "nli14_idf"
    nli_predict_fn = get_nli14_cache_client()
    solve_save_bioclaim_w_nli(nli_predict_fn, run_name, split)


def nli14_no_idf():
    split = "dev"
    run_name = "nli14"
    nli_predict_fn = get_nli14_cache_client()
    solve_save_bioclaim_w_nli(nli_predict_fn, run_name, split, False)


def nli14_clue_idf():
    split = "dev"
    run_name = "nli14_clue"
    nli_predict_fn = get_nli14_cache_client()
    solve_save_bioclaim_w_nli_clue_idf(nli_predict_fn, run_name, split)


def nlits_clue_idf():
    split = "dev"
    run_name = "pep_clue"
    nli_predict_fn = get_pep_cache_client()
    solve_save_bioclaim_w_nli_clue_idf(nli_predict_fn, run_name, split)



def nli14_enum():
    split = "test"
    run_name = "nli14_enum_idf"
    nli_predict_fn = get_nli14_cache_client()
    solve_save_bioclaim_w_nli_enum(nli_predict_fn, run_name, split)


def nlits_enum():
    split = "test"
    run_name = "nlits_enum_idf"
    nli_predict_fn = get_pep_cache_client()
    solve_save_bioclaim_w_nli_enum(nli_predict_fn, run_name, split)


def common_run(split, nli_type, use_idf):
    run_name = f"{split}_{nli_type}" + ("_idf" if use_idf else "")
    if nli_type == "nli":
        nli_predict_fn = get_nli14_cache_client()
    elif nli_type == "nli_pep":
        nli_predict_fn = get_pep_cache_client()
    else:
        assert False

    solve_save_bioclaim_w_nli_enum(nli_predict_fn, run_name, split, use_idf)



def main():
    c_log.setLevel(logging.INFO)
    split = "test"
    todo = [
        # ("nli", False),
        ("nli_pep", False),
        # ("nli_pep", True),
        ("nli_pep", True)
    ]
    for nli_type, use_idf in todo:
        common_run(split, nli_type, use_idf)


if __name__ == "__main__":
    main()
