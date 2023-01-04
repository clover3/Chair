from adhoc.bm25_class import BM25
from dataset_specific.scitail import get_scitail_questions
from trainer_v2.keras_server.name_short_cuts import get_nli14_cache_client, get_pep_cache_client
from trainer_v2.per_project.tli.qa_scorer.bm25_system import build_stats
from trainer_v2.per_project.tli.qa_scorer.nli_token_system import NLIBasedRelevanceMultiSeg
from trainer_v2.per_project.tli.scitail_qa_eval.eval_helper import batch_solve_save_scitail_qa


def common_run(nli_type, use_idf):
    if nli_type == "nli":
        nli_predict_fn = get_nli14_cache_client()
    elif nli_type == "nli_pep":
        nli_predict_fn = get_pep_cache_client()
    else:
        assert False

    run_name = f"{nli_type}" + ("_idf" if use_idf else "")
    if use_idf:
        df, cdf, avdl = build_stats(get_scitail_questions())
        bm25 = BM25(df, avdl=avdl, num_doc=cdf, k1=0.00001, k2=100, b=0.5,
                    drop_stopwords=True)
        module = NLIBasedRelevanceMultiSeg(nli_predict_fn, bm25)
    else:
        module = NLIBasedRelevanceMultiSeg(nli_predict_fn)

    batch_solve_save_scitail_qa(module.batch_predict, run_name)


def main():
    todo = [
        # ("nli", False),
        # ("nli", True),
        ("nli_pep", False),
        ("nli_pep", True)
    ]

    for nli_type, use_idf in todo:
        common_run(nli_type, use_idf)



if __name__ == "__main__":
    main()
