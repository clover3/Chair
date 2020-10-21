from typing import List

from arg.perspectives.runner_qck.qcknc_common import start_generate_jobs_for_sub_split
from arg.perspectives.runner_uni.build_topic_lm import build_gold_lms_for_split, ClaimLM
from arg.qck.decl import QCKQuery, KDP
# Make payload without any annotation
from arg.qck.token_scoring.token_scoring_gen import TokenScoringGen, ScoreVector
from arg.qck.topic_lm.lm_based_scorer import LogOddScorer, ScorerInterface, RawProbabilityScorer
from arg.util import load_run_config


def main():
    config = load_run_config()
    job_prefix = config['job_prefix']
    qk_candidate_name = config['qk_candidate_name']
    score_type = config['score_type']
    split = config['split']
    lms: List[ClaimLM] = build_gold_lms_for_split(split)
    lm_pair_list = list([(str(lm.cid), lm.LM) for lm in lms])

    def get_scorer():
        if score_type == "log_odd":
            return LogOddScorer(lm_pair_list)
        elif score_type == "raw_prob":
            return RawProbabilityScorer(lm_pair_list)
        else:
            assert False

    scorer: ScorerInterface = get_scorer()

    def get_score(q: QCKQuery, kdp: KDP) -> ScoreVector:
        scores = []
        for token in kdp.tokens:
            score = scorer.score_token(q.query_id, token)
            scores.append(score)
        return scores

    start_generate_jobs_for_sub_split(TokenScoringGen(get_score),
                                      qk_candidate_name,
                                      job_prefix,
                                      split)


if __name__ == "__main__":
    main()
