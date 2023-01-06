from trainer_v2.keras_server.name_short_cuts import get_nli14_cache_client
from trainer_v2.per_project.tli.bioclaim_qa.runner.nli_based import get_pep_cache_client, solve_save_bioclaim_w_nli, \
    solve_save_bioclaim_w_nli_clue_idf, solve_save_bioclaim_w_nli_enum


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