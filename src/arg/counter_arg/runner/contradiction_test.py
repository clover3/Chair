import itertools
from typing import List, Callable, NamedTuple

import numpy as np

from arg.counter_arg.contradiction_requester import Request, RequestEx
from arg.counter_arg.eval import collect_failure, EvalCondition
from arg.counter_arg.header import Passage
from arg.counter_arg.methods import bm25_predictor
from arg.counter_arg.methods.bm25_predictor import get_bm25_module
from arg.counter_arg.methods.stance_query import get_stance_check_candidate
from cache import save_to_pickle, load_from_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import NamedNumber
from visualize.html_visual import Cell, HtmlVisualizer


class AnalyzedCase(NamedTuple):
    query: str
    gold: str
    pred: str
    score_g: List
    score_p: List



def ask_contradiction(num):
    split = "training"

    scorer: Callable[[Passage, List[Passage]], List[NamedNumber]] = bm25_predictor.get_scorer(split)

    cont_requester = RequestEx()
    bm25 = get_bm25_module("training")

    def pairwise(query, candidate):
        sent_term_list_q = get_stance_check_candidate(query, bm25)
        sent_term_list_c = get_stance_check_candidate(candidate, bm25)

        r = []
        for sent_q, terms_q in sent_term_list_q:
            for sent_c, terms_c in sent_term_list_c:
                r.append((sent_q, sent_c))
                r.append((sent_c, sent_q))
        return r

    itr = collect_failure(split, scorer, EvalCondition.EntirePortalCounters)
    failure_cases = itertools.islice(itr, num)

    def ask_inst(e):
        query, gold, pred = e

        def ask_cont(target):
            payload = pairwise(query, target)
            return cont_requester.request_multiple(payload)

        r_gold = ask_cont(gold)
        r_pred = ask_cont(pred)
        return AnalyzedCase(
            query=query,
            gold=gold,
            pred=pred,
            score_g=r_gold,
            score_p=r_pred
        )

    analyzed_failture_cases: List[AnalyzedCase] = []
    for a in failure_cases:
        try:
            analyzed_failture_cases.append(ask_inst(a))
        except Exception as e:
            print(e)
    return analyzed_failture_cases


def manual_entry_ask():
    r1 =(' Arts degrees limit opportunities for Universities to offer other courses.',
         ' Despite Prop’s efforts to suggest that there are masses of homeless, would-be engineering students roaming around university campuses, the reality is that universities pack their bankable courses just fine and ensure that they have the capacity to do so.')
    r2 = ('If the value of a degree is judged purely on the likely salary at the end of it, then society has a very real problem.',
          'Of course the financial outcome of doing a degree is of paramount interest to both the student and wider society, suggesting otherwise is sophistry.')
    r3 = (' Celebrities are respected by young people and this is a way in which they can act as a role model and set a positive example.',
          ' It seems, frankly unfair to ask people to destroy their careers on the basis that it will encourage others to do something that the law already requires of them.')

    def swap(pair):
        return (pair[1], pair[0])

    payload = [r1, r2, r3]
    payload_2way = []
    for r in payload:
        payload_2way.append(r)
        payload_2way.append(swap(r))

    cont_requester = Request()

    scores = cont_requester.request_multiple(payload_2way)

    for s1, s2, score in scores:
        print(s1)
        print(s2)
        print(score)


def show_analyzed(analyzed_failture_cases: List[AnalyzedCase]):
    def print_scored_sentences(scores):
        for i, _ in enumerate(scores):
            if i % 2 == 0:
                sent1, sent2, score1, token_score1 = scores[i]
                _, _, score2, token_score2 = scores[i+1]
                if is_cont(score1) or is_cont(score2):
                    print("Sent1: ", sent1)
                    print("Sent2: ", sent2)
                    print("{0:.2f}\t{1:.2f}".format(score1[2], score2[2]))

    def print_analyzed_case(analyzed_case: AnalyzedCase):
        def print_part(score):
            cnt = count_cont(score)
            print("{} of {}".format(cnt, len(score)))
            print_scored_sentences(score)

        print("Gold")
        print_part(analyzed_case.score_g)
        print("Pred")
        print_part(analyzed_case.score_p)


    def is_cont(scores):
        return np.argmax(scores) == 2


    def is_cont_strict(scores):
        return scores[2] > 0.9


    def count_cont(result_list):
        num_cont = sum([1 for _, _, scores, _ in result_list if is_cont(scores)])
        return num_cont

    def count_cont_stric(result_list):
        num_cont = sum([1 for _, _, scores, _ in result_list if is_cont(scores)])
        return num_cont

    def count_cont_pair(result_list):
        cnt = 0
        for i, _ in enumerate(result_list):
            if i % 2 == 0:
                s1 = result_list[i][2]
                s2 = result_list[i+1][2]

                if is_cont(s1) and is_cont(s2):
                    cnt += 1
        return cnt

    c_g_list = list([count_cont(e.score_g) for e in analyzed_failture_cases])
    c_p_list = list([count_cont(e.score_p) for e in analyzed_failture_cases])

    for idx, dp in enumerate(analyzed_failture_cases):
        print("Data point ", idx)
        print("------------")
        print_analyzed_case(dp)

    print(c_g_list)
    print(c_p_list)



def show_analyzed_html(analyzed_failture_cases: List[AnalyzedCase]):
    tokenizer = get_tokenizer()
    html = HtmlVisualizer("ca_contradiction_tokens.html")
    def get_token_scored_cells(sent1, sent2, token_score):
        tokens1 = tokenizer.tokenize(sent1)
        tokens2 = tokenizer.tokenize(sent2)
        print(token_score)

        score_for_1 = token_score[1:1+len(tokens1)]
        score_for_2 = token_score[2 + len(tokens1) : 2 + len(tokens1) + len(tokens2)]

        assert len(tokens1) == len(score_for_1)
        assert len(tokens2) == len(score_for_2)


        def get_cells(tokens, scores):
            cap = max(max(scores), 1)
            factor = 100 / cap
            def normalize_score(s):
                return min(s * factor , 100)
            return list([Cell(t, normalize_score(s)) for t, s in zip(tokens, scores)])

        cells1 = get_cells(tokens1, score_for_1)
        cells2 = get_cells(tokens2, score_for_2)
        return cells1, cells2

    def print_scored_sentences(scores):
        for i, _ in enumerate(scores):
            if i % 2 == 0:
                sent1, sent2, score1, token_score1 = scores[i]
                _, _, score2, token_score2 = scores[i+1]
                if is_cont(score1):
                    cells1, cells2 = get_token_scored_cells(sent1, sent2, token_score1)
                    html.write_paragraph("Forward, P(Contradiction) = {}".format(score1[2]))
                    html.write_table([cells1])
                    html.write_table([cells2])

                if is_cont(score2):
                    cells1, cells2 = get_token_scored_cells(sent2, sent1, token_score2)
                    html.write_paragraph("Backward, P(Contradiction) = {}".format(score2[2]))
                    html.write_table([cells1])
                    html.write_table([cells2])

    def print_analyzed_case(analyzed_case: AnalyzedCase):
        def print_part(score_list):
            cnt = count_cont(score_list)
            s = "{} of {}".format(cnt, len(score_list))
            html.write_paragraph(s)
            print_scored_sentences(score_list)

        html.write_paragraph("Gold")
        print_part(analyzed_case.score_g)
        html.write_paragraph("Pred")
        print_part(analyzed_case.score_p)


    def is_cont(scores):
        return np.argmax(scores) == 2


    def is_cont_strict(scores):
        return scores[2] > 0.9


    def count_cont(result_list):
        num_cont = sum([1 for _, _, scores, _ in result_list if is_cont(scores)])

    for idx, dp in enumerate(analyzed_failture_cases):
        html.write_headline("Data point {}".format(idx))
        html.write_paragraph("------------")
        print_analyzed_case(dp)


def predict_and_save():
    r = ask_contradiction(10)
    save_to_pickle(r, "contradiction_analysis")


def show_result():
    r = load_from_pickle("contradiction_analysis")
    show_analyzed_html(r)


if __name__ == "__main__":
    show_result()
