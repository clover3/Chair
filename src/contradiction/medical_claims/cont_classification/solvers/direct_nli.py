from typing import List, NamedTuple
import numpy as np
from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.cont_classification.defs import ContClassificationSolverIF, ContProblem, \
    ContClassificationProbabilityScorer
from list_lib import lmap
from misc_lib import two_digit_float

from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, get_keras_nli_300_predictor
from visualize.html_visual import HtmlVisualizer


class NLIClassifier(ContClassificationProbabilityScorer):
    def __init__(self,
                 nli_predict_fn: NLIPredictorSig,
                 ):
        self.nli_predict_fn = nli_predict_fn

    def solve_batch(self, problems: List[ContProblem]) -> List[float]:
        def convert(p):
            return p.claim1_text, p.claim2_text

        text_pairs = list(map(convert, problems))
        preds: List[List[float]] = self.nli_predict_fn(text_pairs)
        def get_cont_score(probs):
            return probs[2]

        return list(map(get_cont_score, preds))


def get_nli14_classifier() -> ContClassificationProbabilityScorer:
    return NLIClassifier(get_keras_nli_300_predictor())


class TestComp(NamedTuple):
    name: str
    build_pair_fn: Callable[[ContProblem], Tuple[str,str]]
    score_getter: Callable[[List[float]], float]


def get_c1_c2(p: ContProblem):
    return p.claim1_text, p.claim2_text


def get_c1_q(p: ContProblem):
    question_text = p.question
    return p.claim1_text, question_text


def get_c1_q_yn(word, p: ContProblem):
    question_text = p.question + " " + word
    return p.claim1_text, question_text


def get_c_q_add_word(seg_no, word):
    def get_pair(p: ContProblem):
        question_text = p.question + " " + word
        prem = [p.claim1_text, p.claim2_text][seg_no]
        return prem, question_text

    return get_pair


def get_c2_q(p: ContProblem):
    question_text = p.question
    return p.claim2_text, question_text


def get_entail(probs):
    return probs[0]


def get_neutral(probs):
    return probs[1]


def get_cont(probs):
    return probs[2]


def get_one_minus_neutral(probs):
    return 1 - probs[1]


class NLIWQuestion(ContClassificationProbabilityScorer):
    def __init__(self,
                 nli_predict_fn: NLIPredictorSig,
                 ):
        self.nli_predict_fn = nli_predict_fn

    def solve_batch(self, problems: List[ContProblem]) -> List[float]:
        # Good pair
        #   entail(t1, q) / cont(t2, q) / cont(t1, t2)
        #   cont(t1, q) / entail(t2, q) / cont(t1, t2)
        #   not neutral(t1, q) / not neutral(t2, q) / cont(t1, t2)
        test_fns = [
            TestComp("not neutral(t1, q)", get_c1_q, get_one_minus_neutral),
            TestComp("not neutral(t2, q)", get_c2_q, get_one_minus_neutral),
            TestComp("cont(t1, t2)", get_c1_c2, get_cont),
        ]

        def get_pair_payloads(p: ContProblem):
            for test_comp in test_fns:
                yield test_comp.build_pair_fn(p)

        payloads = []
        for p in problems:
            payloads.extend(get_pair_payloads(p))

        preds: List[List[float]] = self.nli_predict_fn(payloads)
        preds_d: Dict[Tuple[str, str], List[float]] = dict(zip(payloads, preds))

        out_scores = []
        n_test = len(test_fns)
        for p in problems:
            score_sum = 0
            for test_comp in test_fns:
                pred = preds_d[test_comp.build_pair_fn(p)]
                score = test_comp.score_getter(pred)
                score_sum += score
            out_scores.append(score_sum / n_test)

        return out_scores


class NLITestComps(ContClassificationProbabilityScorer):
    def __init__(self,
                 nli_predict_fn: NLIPredictorSig,
                 test_comps: List[TestComp],
                 score_combine: Callable[[Dict[str, float]], float],
                 ):
        self.nli_predict_fn = nli_predict_fn
        self.test_comps: List[TestComp] = test_comps
        self.score_combine = score_combine

    def solve_batch(self, problems: List[ContProblem]) -> List[float]:
        # Good pair
        #   entail(t1, q) / cont(t2, q) / cont(t1, t2)
        #   cont(t1, q) / entail(t2, q) / cont(t1, t2)
        #   not neutral(t1, q) / not neutral(t2, q) / cont(t1, t2)

        def get_pair_payloads(p: ContProblem):
            for test_comp in self.test_comps:
                yield test_comp.build_pair_fn(p)

        payloads = []
        for p in problems:
            payloads.extend(get_pair_payloads(p))

        payloads = list(set(payloads))
        preds: List[List[float]] = self.nli_predict_fn(payloads)
        preds_d: Dict[Tuple[str, str], List[float]] = dict(zip(payloads, preds))

        out_scores = []
        for p in problems:
            score_d = {}
            for test_comp in self.test_comps:
                pred = preds_d[test_comp.build_pair_fn(p)]
                score = test_comp.score_getter(pred)
                score_d[test_comp.name] = score

            combined_score = self.score_combine(score_d)
            out_scores.append(combined_score)
        return out_scores


class NLIQ3Debug(ContClassificationProbabilityScorer):
    def __init__(self,
                 nli_predict_fn: NLIPredictorSig,
                 ):
        self.nli_predict_fn = nli_predict_fn

    def solve_batch(self, problems: List[ContProblem]) -> List[float]:
        # Good pair
        #   entail(t1, q) / cont(t2, q) / cont(t1, t2)
        #   cont(t1, q) / entail(t2, q) / cont(t1, t2)
        #   not neutral(t1, q) / not neutral(t2, q) / cont(t1, t2)
        html = HtmlVisualizer("direct_nli.html")
        test_fns = [
            TestComp("entail(t1, q+ys)", get_c_q_add_word(0, "? Yes"), get_entail),
            TestComp("entail(t2, q+no)", get_c_q_add_word(1, "? No"), get_entail),
            TestComp("entail(t1, q+no)", get_c_q_add_word(0, "? No"), get_entail),
            TestComp("entail(t2, q+ys)", get_c_q_add_word(1, "? Yes"), get_entail),

            TestComp("cont(t1, q+no)", get_c_q_add_word(0, "? No"), get_cont),
            TestComp("cont(t2, q+ys)", get_c_q_add_word(1, "? Yes"), get_cont),
            TestComp("cont(t1, q+ys)", get_c_q_add_word(0, "? Yes"), get_cont),
            TestComp("cont(t2, q+no)", get_c_q_add_word(1, "? No"), get_cont),

            TestComp("cont(t1, t2)", get_c1_c2, get_cont),
        ]

        def get_pair_payloads(p: ContProblem):
            for test_comp in test_fns:
                yield test_comp.build_pair_fn(p)

        payloads = []
        for p in problems:
            payloads.extend(get_pair_payloads(p))

        payloads = list(set(payloads))
        preds: List[List[float]] = self.nli_predict_fn(payloads)
        preds_d: Dict[Tuple[str, str], List[float]] = dict(zip(payloads, preds))

        def discretize(score_d):
            new_d = {}
            for k, v in score_d.items():
                if v >= 0.5:
                    new_d[k] = True
                else:
                    new_d[k] = False
            return new_d

        def combine(score_d):
            new_d = discretize(score_d)
            opt1 = new_d["entail(t1, q+ys)"] and new_d["entail(t2, q+no)"]
            opt2 = new_d["entail(t1, q+no)"] and new_d["entail(t2, q+ys)"]
            opt3 = new_d["cont(t1, q+no)"] and new_d["cont(t2, q+ys)"]
            opt4 = new_d["cont(t1, q+ys)"] and new_d["cont(t2, q+no)"]
            condition1 = max(opt1, opt2, opt3, opt4)
            condition2 = new_d["cont(t1, t2)"]
            return condition1 or condition2


        html_write = html.write_paragraph
        out_scores = []
        for p in problems:
            html_write("Question: " + p.question)
            html_write("Claim 1: " + p.claim1_text)
            html_write("Claim 2: " + p.claim2_text)
            html_write("Label: {}".format(p.label))
            score_d = {}
            for test_comp in test_fns:
                pred = preds_d[test_comp.build_pair_fn(p)]
                score = test_comp.score_getter(pred)
                score_d[test_comp.name] = score
                pred_str = ", ".join(lmap(two_digit_float, pred))
                pred_label = "ENC"[np.argmax(pred)]
                html_write("{0} {1:.2f} {2} {3}".format(test_comp.name, score, pred_str, pred_label))
            new_d = discretize(score_d)
            opt1 = new_d["entail(t1, q+ys)"] * new_d["entail(t2, q+no)"]
            opt2 = new_d["entail(t1, q+no)"] * new_d["entail(t2, q+ys)"]
            opt3 = new_d["cont(t1, q+no)"] * new_d["cont(t2, q+ys)"]
            opt4 = new_d["cont(t1, q+ys)"] * new_d["cont(t2, q+no)"]
            condition1 = max(opt1, opt2, opt3, opt4)
            condition2 = new_d["cont(t1, t2)"]
            condition1_list = ", ".join(map(two_digit_float, [opt1, opt2, opt3, opt3]))
            html_write("Per condition score: {}".format(condition1_list))
            combined_score = combine(score_d)
            html_write("Classification Pred: {0} ({1:.2f})".format(combined_score >= 0.5, combined_score))
            out_scores.append(combined_score)
            html.write_bar()
        return out_scores


def do_average(scores: Dict[str, float]):
    return sum(scores.values()) / len(scores)


def get_nli_q1() -> NLITestComps:
    test_fns = [
        TestComp("not neutral(t1, q)", get_c1_q, get_one_minus_neutral),
        TestComp("not neutral(t2, q)", get_c2_q, get_one_minus_neutral),
        TestComp("cont(t1, t2)", get_c1_c2, get_cont),
    ]
    return NLITestComps(get_keras_nli_300_predictor(), test_fns, do_average)


def get_nli_q2() -> NLITestComps:
    test_fns = [
        TestComp("entail(t1, q+ys)", get_c_q_add_word(0, "Yes"), get_entail),
        TestComp("entail(t2, q+no)", get_c_q_add_word(1, "No"), get_entail),
        TestComp("entail(t1, q+no)", get_c_q_add_word(0, "No"), get_entail),
        TestComp("entail(t2, q+ys)", get_c_q_add_word(1, "Yes"), get_entail),

        TestComp("cont(t1, q+no)", get_c_q_add_word(0, "No"), get_cont),
        TestComp("cont(t2, q+ys)", get_c_q_add_word(1, "Yes"), get_cont),
        TestComp("cont(t1, q+ys)", get_c_q_add_word(0, "Yes"), get_cont),
        TestComp("cont(t2, q+no)", get_c_q_add_word(1, "No"), get_cont),

        TestComp("cont(t1, t2)", get_c1_c2, get_cont),
    ]
    def combine(score_d):
        opt1 = score_d["entail(t1, q+ys)"] + score_d["entail(t2, q+no)"]
        opt2 = score_d["entail(t1, q+no)"] + score_d["entail(t2, q+ys)"]
        opt3 = score_d["cont(t1, q+no)"] + score_d["cont(t2, q+ys)"]
        opt4 = score_d["cont(t1, q+ys)"] + score_d["cont(t2, q+no)"]

        condition1 = max(opt1, opt2, opt3, opt4)
        condition2 = score_d["cont(t1, t2)"]
        return (condition1 + condition2) / 2

    return NLITestComps(get_keras_nli_300_predictor(), test_fns, combine)


def get_nli_q3() -> NLITestComps:

    test_fns = [
        TestComp("entail(t1, q+ys)", get_c_q_add_word(0, "? Yes"), get_entail),
        TestComp("entail(t2, q+no)", get_c_q_add_word(1, "? No"), get_entail),
        TestComp("entail(t1, q+no)", get_c_q_add_word(0, "? No"), get_entail),
        TestComp("entail(t2, q+ys)", get_c_q_add_word(1, "? Yes"), get_entail),

        TestComp("cont(t1, q+no)", get_c_q_add_word(0, "? No"), get_cont),
        TestComp("cont(t2, q+ys)", get_c_q_add_word(1, "? Yes"), get_cont),
        TestComp("cont(t1, q+ys)", get_c_q_add_word(0, "? Yes"), get_cont),
        TestComp("cont(t2, q+no)", get_c_q_add_word(1, "? No"), get_cont),

        TestComp("cont(t1, t2)", get_c1_c2, get_cont),
    ]
    do_rest_html = True
    def combine(score_d):
        opt1 = score_d["entail(t1, q+ys)"] + score_d["entail(t2, q+no)"]
        opt2 = score_d["entail(t1, q+no)"] + score_d["entail(t2, q+ys)"]
        opt3 = score_d["cont(t1, q+no)"] + score_d["cont(t2, q+ys)"]
        opt4 = score_d["cont(t1, q+ys)"] + score_d["cont(t2, q+no)"]
        condition1 = max(opt1, opt2, opt3, opt4)
        condition2 = score_d["cont(t1, t2)"]
        return (condition1 + condition2) / 2

    return NLITestComps(get_keras_nli_300_predictor(), test_fns, combine)


def get_nli_q3_debug():
    return NLIQ3Debug(get_keras_nli_300_predictor())


def get_nli_q4() -> NLITestComps:
    test_fns = [
        # IF YES/NO
        TestComp("entail(t1, q+ys)", get_c_q_add_word(0, "? Yes"), get_entail),
        TestComp("entail(t2, q+no)", get_c_q_add_word(1, "? No"), get_entail),
        # IF YES/NO
        TestComp("entail(t1, q+no)", get_c_q_add_word(0, "? No"), get_entail),
        TestComp("entail(t2, q+ys)", get_c_q_add_word(1, "? Yes"), get_entail),

        # IF YES/NO
        TestComp("cont(t1, q+no)", get_c_q_add_word(0, "? No"), get_cont),
        TestComp("cont(t2, q+ys)", get_c_q_add_word(1, "? Yes"), get_cont),
        TestComp("cont(t1, q+ys)", get_c_q_add_word(0, "? Yes"), get_cont),
        TestComp("cont(t2, q+no)", get_c_q_add_word(1, "? No"), get_cont),
    ]
    def combine(score_d):
        t1_is_yes = max(score_d["entail(t1, q+ys)"], score_d["cont(t1, q+no)"])
        t1_is_no = max(score_d["entail(t1, q+no)"], score_d["cont(t1, q+ys)"])

        t2_is_no = max(score_d["entail(t2, q+no)"], score_d["cont(t2, q+ys)"])
        t2_is_yes = max(score_d["entail(t2, q+ys)"], score_d["cont(t2, q+no)"])

        opt1 = (t1_is_yes + t2_is_no) / 2
        opt2 = (t1_is_no + t2_is_yes) / 2
        condition1 = max(opt1, opt2)
        return condition1

    return NLITestComps(get_keras_nli_300_predictor(), test_fns, combine)
