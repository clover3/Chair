from alignment import MatrixScorerIF
from alignment.matrix_scorers.methods.all_nothing_scorer import AllOneScorer, AllZeroScorer
from alignment.matrix_scorers.methods.ensemble_scorer import EnsembleScorer
from alignment.matrix_scorers.methods.exact_match_scorer import SegmentExactMatchScorer, TokenExactMatchScorer
from alignment.matrix_scorers.methods.get_word2vec_scorer import get_word2vec_scorer_from_d
from alignment.matrix_scorers.methods.random_score import RandomScorer


def get_scorer(scorer_name) -> MatrixScorerIF:
    if scorer_name == "lexical_v1":
        s1 = SegmentExactMatchScorer()
        s2 = get_word2vec_scorer_from_d()
        scorer = EnsembleScorer([s1, s2])
    elif scorer_name == "random":
        scorer: MatrixScorerIF = RandomScorer()
    elif scorer_name == "segment_exact_match":
        scorer: MatrixScorerIF = SegmentExactMatchScorer()
    elif scorer_name == "token_exact_match":
        scorer: MatrixScorerIF = TokenExactMatchScorer()
    elif scorer_name == "all_one":
        scorer: MatrixScorerIF = AllOneScorer()
    elif scorer_name == "all_zero":
        scorer: MatrixScorerIF = AllZeroScorer()
    else:
        raise ValueError

    return scorer