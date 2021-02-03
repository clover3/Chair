import sys
from collections import Counter
from typing import List

import numpy as np

from cache import load_from_pickle
from explain.genex.load import load_packed, PackedInstance
from models.classic.stopword import load_stopwords_for_query


class Config1:
    drop_stopwords = True
    max_terms = 5
    reorder = False


def get_answer_maker(config):
    stopwords = load_stopwords_for_query()

    def make_answer1(problem: PackedInstance, score: np.array) -> List[str]:
        # among tokens from documents
        # select unique words that has highest score
        token_score = Counter()
        n_appear = Counter()
        max_len = len(problem.input_ids)
        for idx in range(max_len):
            if problem.input_mask[idx] == 0:
                break

            # skip query tokens
            if problem.segment_ids[idx] == 0:
                continue

            token_idx = problem.idx_mapping[idx]
            token = problem.word_tokens[token_idx]
            token_score[token] += score[idx]
            n_appear[token] += 1

        print(" ".join(problem.word_tokens))
        out_tokens = []
        max_score = None
        for token, token_score in token_score.most_common():
            if len(out_tokens) > config.max_terms:
                break
            if config.drop_stopwords and token in stopwords:
                continue

            if max_score is None:
                max_score = token_score
                score_cut = max_score * 0.1

            if len(out_tokens) == 0:
                include = True
            else:
                if token_score > score_cut:
                    include = True
                else:
                    include = False

            print("{0} {1:.4f} {2} {3}".format(token, token_score, n_appear[token], include))
            if include:
                out_tokens.append(token)
            else:
                break
        return out_tokens
    return make_answer1


def main():
    data_name = sys.argv[1]
    score_name = sys.argv[2]
    save_path = sys.argv[3]

    scores: List[np.array] = load_from_pickle(score_name)
    data: List[PackedInstance] = load_packed(data_name)

    make_answer = get_answer_maker(Config1)

    out_f = open(save_path, 'w')
    for problem, score in zip(data, scores):
        answer_tokens: List[str] = make_answer(problem, score)
        answer = " ".join(answer_tokens)
        out_f.write(answer + "\n")
    out_f.close()


if __name__ == "__main__":
    main()