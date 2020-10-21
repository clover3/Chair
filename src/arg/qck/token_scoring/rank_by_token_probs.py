from collections import defaultdict
from typing import List, Dict, Tuple, Set

import math
import numpy as np
from krovetzstemmer import Stemmer

from arg.perspectives.qck.qcknc_datagen import get_eval_candidates_as_qck, get_eval_candidates_1k_as_qck
from arg.qck.decl import QCKCandidate
from arg.qck.token_scoring.collect_score import WordAsID, ids_to_word_as_id, decode_word_as_id
from arg.qck.token_scoring.decl import TokenScore
from cache import load_pickle_from
from data_generator.tokenizer_wo_tf import get_tokenizer, get_word_level_location, pretty_tokens
from evals.trec import scores_to_ranked_list_entries, write_trec_ranked_list_entry
from exec_lib import run_func_with_config
from list_lib import lmap
from misc_lib import get_second, average
from models.classic.stopword import load_stopwords_ex


class Scorer:
    def __init__(self, d: Dict[WordAsID, np.array], skip_stopwords=True, stem=True):
        self.tokenizer = get_tokenizer()

        self.stopwords_as_ids: Set[WordAsID] = set()
        new_d = {}
        if skip_stopwords:
            stopwords = load_stopwords_ex()
            for key in d.keys():
                tokens = decode_word_as_id(self.tokenizer, key)
                if len(tokens) == 1 and tokens[0] in stopwords:
                    pass
                    self.stopwords_as_ids.add(key)
                else:
                    new_d[key] = d[key]
            d = new_d

        if stem:
            d_raw = defaultdict(list)
            stemmer = Stemmer()

            for key in d.keys():
                tokens = decode_word_as_id(self.tokenizer, key)
                plain_word = pretty_tokens(tokens, True)
                stemmed = stemmer.stem(plain_word)
                d_raw[stemmed].append(d[key])

            new_d: Dict[str, TokenScore] = {}
            for key, items in d_raw.items():
                score: TokenScore = [average([t[0] for t in items]), average([t[1] for t in items])]
                new_d[key] = score
            d = new_d
            self.stem = True
            self.stemmer = stemmer
            self.log_odd = self.log_odd_w_stem

        self.d = d
        self.smoothing = 0.1

    def log_odd_w_stem(self, word: WordAsID) -> float:
        tokens = decode_word_as_id(self.tokenizer, word)
        plain_word: str = pretty_tokens(tokens, True)
        stemmed: str = self.stemmer.stem(plain_word)
        if stemmed in self.d:
            token_score: TokenScore = self.d[stemmed]
            p_pos = token_score[0]
            p_neg = token_score[1]
            if not (p_pos >= 0):
                print(len(word))
                print(word, token_score)
            assert p_pos >= 0
            assert p_neg >= 0
            eps = 1e-10
            return math.log(p_pos+eps) - math.log(p_neg+eps)
        else:
            return 0

    def log_odd(self, word: WordAsID) -> float:
        eps = 1e-10
        if word in self.d:
            token_score: TokenScore = self.d[word]
            p_pos = token_score[0]
            p_neg = token_score[1]
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
            print(tokens)
            print(ids)
            print(intervals)
            print(word_scores)

        rationale = ""
        for word_idx, (st, ed) in enumerate(intervals):
            rationale += " {0} ({1:.1f})".format("".join(tokens[st:ed]), word_scores[word_idx])
        # print(text, rationale)
        return sum(word_scores)


def main(config):
    split = config['split']
    top_k = config['top_k']
    word_prob_path = config['word_prob_path']
    run_name = config['run_name']
    save_path = config['save_path']
    if top_k == 50:
        candidate_d: Dict[str, List[QCKCandidate]] = get_eval_candidates_as_qck(split)
    elif top_k == 1000:
        candidate_d: Dict[str, List[QCKCandidate]] = get_eval_candidates_1k_as_qck(split)
    else:
        assert False

    per_query_infos: Dict[str, Dict[WordAsID, np.array]] = load_pickle_from(word_prob_path)

    all_ranked_list_entries = []

    for query_id, d in per_query_infos.items():
        scorer = Scorer(d, True)
        candidates: List[QCKCandidate] = candidate_d[query_id]

        entries = []
        for c in candidates:
            e = c.id, scorer.score(c.text)
            entries.append(e)
        entries.sort(key=get_second, reverse=True)

        ranked_list_entries = scores_to_ranked_list_entries(entries, run_name, query_id)
        all_ranked_list_entries.extend(ranked_list_entries)

    write_trec_ranked_list_entry(all_ranked_list_entries, save_path)


if __name__ == "__main__":
    run_func_with_config(main)
