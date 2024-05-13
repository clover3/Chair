from adhoc.resource.bm25t_method_loader import get_bm25t, is_bm25t_method
from adhoc.resource.qlt_method_loader import is_qlt_method, get_ql, get_qlt


class RerankScorerWrap:
    def __init__(self, score_fn, is_neural=False, batch_size=None):
        self.score_fn = score_fn
        self.is_neural = is_neural
        if self.is_neural:
            self.batch_size = 64
        else:
            self.batch_size = 1

        if batch_size is not None:
            self.batch_size = batch_size

    def get_outer_batch_size(self):
        return self.batch_size


# Actual implementations should be loaded locally


def get_rerank_scorer(method: str) -> RerankScorerWrap:
    if is_bm25t_method(method):
        score_fn = get_bm25t(method)
        rerank_scorer = RerankScorerWrap(score_fn, False)
    elif method == "ql":
        score_fn = get_ql(method)
        rerank_scorer = RerankScorerWrap(score_fn, False)
    elif is_qlt_method(method):
        score_fn = get_qlt(method)
        rerank_scorer = RerankScorerWrap(score_fn, False)
    elif method == "ce_msmarco_mini_lm" or method == "ce":
        from ptorch.cross_encoder.get_ce_msmarco_mini_lm import get_ce_msmarco_mini_lm_score_fn
        score_fn = get_ce_msmarco_mini_lm_score_fn()
        rerank_scorer = RerankScorerWrap(score_fn, True)
    elif method == "splade":
        # from trainer_v2.per_project.transparency.mmp.runner.splade_predict import get_local_xmlrpc_scorer_fn
        # score_fn = get_local_xmlrpc_scorer_fn()
        from ptorch.try_public_models.splade import get_splade_as_reranker
        score_fn = get_splade_as_reranker()
        rerank_scorer = RerankScorerWrap(score_fn, True)
    elif method == "contriever":
        from ptorch.try_public_models.contriever import get_contriever_as_reranker
        score_fn = get_contriever_as_reranker()
        rerank_scorer = RerankScorerWrap(score_fn, True)
    elif method == "contriever-msmarco":
        from ptorch.try_public_models.contriever import get_contriever_as_reranker
        score_fn = get_contriever_as_reranker("facebook/contriever-msmarco")
        rerank_scorer = RerankScorerWrap(score_fn, True)
    elif method == "tas_b":
        from ptorch.try_public_models.tas_b import get_tas_b_as_reranker
        score_fn = get_tas_b_as_reranker()
        rerank_scorer = RerankScorerWrap(score_fn, True)
    elif method.startswith("pepn_"):
        from ptorch.pep.get_pep_nseg_scoring import get_pepn_score_fn_auto
        score_fn = get_pepn_score_fn_auto()
        rerank_scorer = RerankScorerWrap(score_fn, True, 10000)
    else:
        raise ValueError(f"Method {method} is not expected" )

    return rerank_scorer
