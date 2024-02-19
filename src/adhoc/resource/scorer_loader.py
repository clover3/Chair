from adhoc.resource.bm25t_method_loader import get_bm25t, is_bm25t_method


class RerankScorerWrap:
    def __init__(self, score_fn, is_neural=False):
        self.score_fn = score_fn
        self.is_neural = is_neural

    def get_outer_batch_size(self):
        if self.is_neural:
            return 64
        else:  # Assume neural method
            return 1  #000 * 1000


# Actual implementations should be loaded locally


def get_rerank_scorer(method: str) -> RerankScorerWrap:
    if is_bm25t_method(method):
        score_fn = get_bm25t(method)
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
    else:
        raise ValueError(f"Method {method} is not expected" )

    return rerank_scorer
