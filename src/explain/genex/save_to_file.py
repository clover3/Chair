import random
import sys
from collections import Counter
from typing import List, Dict

import numpy as np

from cache import load_from_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.genex.load import load_packed, PackedInstance
from misc_lib import recover_int_list_str
from models.classic.stopword import load_stopwords_for_query


class DropStop:
    drop_stopwords = True
    max_terms = 10
    reorder = False
    cut_factor = 0.1
    name = "1a"


class RandomConfig:
    drop_stopwords = True
    max_terms = 10
    reorder = False
    cut_factor = 0.001
    name = "1a"


class Config2:
    drop_stopwords = False
    max_terms = 10
    reorder = False
    cut_factor = 0.1
    name = "1b"


class ConfigShort:
    drop_stopwords = True
    max_terms = 10
    reorder = False
    cut_factor = 0.5
    name = "2a"


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
            if token_idx == -1:
                assert problem.word_tokens[token_idx] == "[CLS]"
                print("skip cls token")
            token = problem.word_tokens[token_idx]
            token_score[token] += score[idx]
            n_appear[token] += 1

        out_tokens = []
        max_score = None
        for token, token_score in token_score.most_common():
            if len(out_tokens) > config.max_terms:
                break
            if config.drop_stopwords and token in stopwords:
                continue

            if max_score is None:
                max_score = token_score
                score_cut = max_score * config.cut_factor

            if len(out_tokens) == 0:
                include = True
            else:
                if token_score > score_cut:
                    include = True
                else:
                    include = False

            if include:
                out_tokens.append(token)
            else:
                break
        return out_tokens
    return make_answer1


def get_term(tokens) -> str:
    s = ""
    for t in tokens:
        if t.startswith("##"):
            t = t[2:]
        s += t

    return s


def get_answer_maker_term_level(config):
    stopwords = load_stopwords_for_query()
    stopwords.add("[SEP]")
    stopwords.add("[CLS]")
    tokenizer = get_tokenizer()

    def make_answer1(problem: PackedInstance, score: Dict) -> List[str]:
        # among tokens from documents
        # select unique words that has highest score
        out_tokens = []
        max_score = None
        for term_id_str, term_score in Counter(score).most_common():
            if len(out_tokens) > config.max_terms:
                break

            input_ids = recover_int_list_str(term_id_str)
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            term = get_term(tokens)
            if config.drop_stopwords and term in stopwords:
                continue

            if max_score is None:
                max_score = term_score
                score_cut = max_score * config.cut_factor

            if len(out_tokens) == 0:
                include = True
            else:
                if term_score > score_cut:
                    include = True
                else:
                    include = False

            if include:
                out_tokens.append(term)
            else:
                break
        return out_tokens
    return make_answer1


def get_answer_maker_token_level(config):
    stopwords = load_stopwords_for_query()

    def make_answer(problem: str, score: np.array) -> List[str]:
        tokens = problem.split()
        sep_idx = tokens.index("[SEP]")
        # among tokens from documents
        # select unique words that has highest score
        token_score = Counter()
        n_appear = Counter()
        max_len = len(tokens)
        print(tokens)
        print(max_len)
        print(len(score))
        for idx in range(sep_idx + 1, max_len):
            # skip query tokens
            if tokens[idx] == "[PAD]":
                break
            token = tokens[idx]
            token_score[token] += score[idx]
            n_appear[token] += 1

        out_tokens = []
        max_score = None
        for token, token_score in token_score.most_common():
            if len(out_tokens) > config.max_terms:
                break
            if config.drop_stopwords and token in stopwords:
                continue

            if max_score is None:
                max_score = token_score
                score_cut = max_score * config.cut_factor

            if len(out_tokens) == 0:
                include = True
            else:
                if token_score > score_cut:
                    include = True
                else:
                    include = False

            if include:
                out_tokens.append(token)
            else:
                break
        return out_tokens
    return make_answer


def query_as_answer(problem: PackedInstance) -> List[str]:
    sep_idx = problem.word_tokens.index("[SEP]")
    q_terms = problem.word_tokens[:sep_idx]
    return q_terms


def random_answer(problem: PackedInstance) -> List[str]:
    all_tokens = problem.word_tokens
    random.shuffle(all_tokens)
    return all_tokens[:10]


def main():
    data_name = sys.argv[1]
    score_name = sys.argv[2]
    save_path = sys.argv[3]

    scores: List[np.array] = load_from_pickle(score_name)
    data: List[PackedInstance] = load_packed(data_name)

    save_score_to_file(data, DropStop, save_path, scores)


def save_score_to_file(data, config, save_path, scores):
    make_answer = get_answer_maker(config)
    out_f = open(save_path, 'w')
    for problem, score in zip(data, scores):
        answer_tokens: List[str] = make_answer(problem, score)
        answer = " ".join(answer_tokens)
        out_f.write(answer + "\n")
    out_f.close()


def save_score_to_file_term_level(data, config, save_path, scores):
    make_answer = get_answer_maker_term_level(config)
    out_f = open(save_path, 'w')
    for problem, score in zip(data, scores):
        answer_tokens: List[str] = make_answer(problem, score)
        answer = " ".join(answer_tokens)
        out_f.write(answer + "\n")
    out_f.close()


def save_score_to_answer2(data, save_path, make_answer):
    out_f = open(save_path, 'w')
    for problem in data:
        answer_tokens: List[str] = make_answer(problem)
        answer = " ".join(answer_tokens)
        out_f.write(answer + "\n")
    out_f.close()




if __name__ == "__main__":
    main()