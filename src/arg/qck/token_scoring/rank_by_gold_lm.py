from typing import List, Dict, Tuple

import math
import numpy as np

from arg.perspectives.qck.qcknc_datagen import get_eval_candidates_as_qck, get_eval_candidates_1k_as_qck
from arg.perspectives.runner_uni.build_topic_lm import build_gold_lms_for_split
from arg.qck.decl import QCKCandidate
from arg.qck.token_scoring.collect_score import WordAsID, ids_to_word_as_id
from arg.qck.token_scoring.decl import TokenScore
from arg.qck.topic_lm.lm_based_scorer import RawProbabilityScorer
from data_generator.tokenizer_wo_tf import get_tokenizer, get_word_level_location
from trec.trec_parse import scores_to_ranked_list_entries, write_trec_ranked_list_entry
from exec_lib import run_func_with_config
from list_lib import lmap, left
from misc_lib import get_second


class Scorer:
    def __init__(self, d: Dict[WordAsID, np.array]):
        self.tokenizer = get_tokenizer()
        self.d = d
        self.smoothing = 0.1

    def log_odd(self, word: WordAsID) -> float:
        eps = 1e-10
        if word in self.d:
            token_score: TokenScore = self.d[word]
            p_pos = token_score[1]
            p_neg = token_score[0]
            if not (p_pos >= 0):
                print(len(word))
                print(word, token_score)
            assert p_pos >= 0
            assert p_neg >= 0
            return math.log(p_pos+eps) - math.log(p_neg+eps)
        else:
            return 0

    def score(self, text) -> float:
        tokens = self.tokenizer.tokenize(text)
        ids: List[int] = self.tokenizer.convert_tokens_to_ids(tokens)
        intervals: List[Tuple[int, int]] = get_word_level_location(self.tokenizer, ids)
        words: List[WordAsID] = list([ids_to_word_as_id(ids[st:ed]) for st, ed in intervals])

        word_scores = lmap(self.log_odd, words)
        if not any(word_scores):
            print("WARNING all scores are zero for :", text)
            print(word_scores)

        rationale = ""
        for word_idx, (st, ed) in enumerate(intervals):
            rationale += " {0} ({1:.1f})".format("".join(tokens[st:ed]), word_scores[word_idx])
        print(text, rationale)
        return sum(word_scores)


def main(config):
    split = config['split']
    top_k = config['top_k']
    run_name = config['run_name']
    save_path = config['save_path']
    if top_k == 50:
        candidate_d: Dict[str, List[QCKCandidate]] = get_eval_candidates_as_qck(split)
    elif top_k == 1000:
        candidate_d: Dict[str, List[QCKCandidate]] = get_eval_candidates_1k_as_qck(split)
    else:
        assert False

    lms = build_gold_lms_for_split("val")
    lm_pair_list = list([(str(lm.cid), lm.LM) for lm in lms])
    scorer = RawProbabilityScorer(lm_pair_list)

    all_ranked_list_entries = []
    eps = 1e-10
    for query_id in left(lm_pair_list):
        candidates: List[QCKCandidate] = candidate_d[query_id]
        entries = []
        for c in candidates:
            scores = scorer.score_text(query_id, c.text)
            score = 0
            for token_score in scores:
                pos_prob = math.log(token_score[0]+eps)
                neg_prob = math.log(token_score[1]+eps)
                score += pos_prob - neg_prob
            e = c.id, score
            entries.append(e)
        entries.sort(key=get_second, reverse=True)

        ranked_list_entries = scores_to_ranked_list_entries(entries, run_name, query_id)
        all_ranked_list_entries.extend(ranked_list_entries)

    write_trec_ranked_list_entry(all_ranked_list_entries, save_path)


if __name__ == "__main__":
    run_func_with_config(main)
