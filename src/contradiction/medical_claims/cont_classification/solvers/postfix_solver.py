import math
from typing import List, Callable, Dict, Tuple
import numpy as np

from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from contradiction.medical_claims.cont_classification.defs import ContClassificationProbabilityScorer, ContProblem
from contradiction.medical_claims.cont_classification.solvers.direct_nli import TestComp, get_c_q_add_word, get_entail, \
    get_cont, get_c1_c2
from list_lib import right
from trainer_v2.per_project.tli.bioclaim_qa.eval_helper import get_bioclaim_retrieval_corpus
from trainer_v2.per_project.tli.bioclaim_qa.runner.nli_based import get_pep_cache_client
from trainer_v2.per_project.tli.qa_scorer.bm25_system import build_stats
from trainer_v2.per_project.tli.tli_visualize import til_to_table
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_1, enum_subseq_136, enum_subseq_136_ex
from trainer_v2.per_project.tli.token_level_inference import Numpy2D, Numpy1D, TokenLevelInference, \
    TokenLevelInferenceExclusion, mask_to_str, nc_max_e_avg_reduce_then_softmax
from explain.pairing.run_visualizer.show_cls_probe import NLIVisualize
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig
import scipy.special

from visualize.html_visual import HtmlVisualizer

#    For each problem
#        Enum sub-predictions
#           For each prediction, check if neutral decision is valid
#           The decision is valid if the neutral causing tokens indicates 'topics'
#               What are not topic:  ("we found that", "compared to placebo",
#               High IDF -> delete that token and try again.

def get_neutral_reasons(tli):
    pred = np.argmax(tli, axis=1)

    out_indices = []
    for i in range(len(pred)):
        if pred[i]:
            out_indices.append(i)

    return out_indices


class PostFixModel(ContClassificationProbabilityScorer):
    def __init__(self,
                 nli_predict_fn: NLIPredictorSig,
                 enum_subseq: Callable,
                 test_comps: List[TestComp],
                 score_combine: Callable[[Dict[str, float]], float],
                 is_important_fn: Callable[[str], bool]
                 ):
        self.nli_predict_fn = nli_predict_fn
        self.tli_module = TokenLevelInference(nli_predict_fn, enum_subseq)
        self.test_fns = test_comps
        self.score_combine = score_combine
        self.is_important_fn = is_important_fn

    def solve_batch(self, problems: List[ContProblem]) -> List[float]:
        out_scores = []
        direct_payloads = []
        tli_payloads = []

        for p in problems:
            for test_comp in self.test_fns:
                t1, t2 = test_comp.build_pair_fn(p)
                direct_payloads.append((t1, t2))
                tli_payloads.append((t1, t2))

        preds = self.nli_predict_fn(direct_payloads)
        preds_d = dict(zip(direct_payloads, preds))

        tli_d: Dict[Tuple[str, str], Numpy2D] = self.tli_module.do_batch_return_dict(tli_payloads)
        mod_mapping = {}
        mod_payload = []

        for p in problems:
            # Check if it is neutral for important reason.
            for test_comp in self.test_fns:
                t1, t2 = test_comp.build_pair_fn(p)
                pred = preds_d[(t1, t2)]
                is_neutral = np.argmax(pred) == 1
                if is_neutral:
                    tli: Numpy2D = tli_d[(t1, t2)]
                    not_important_indices = self.get_not_important_indices(tli, t1, t2)
                    if not_important_indices:
                        t2_mod = remove_indices(t2, not_important_indices)
                        mod_payload.append((t1, t2_mod))
                        mod_mapping[(t1, t2)] = t1, t2_mod

        mod_preds = self.nli_predict_fn(mod_payload)
        mod_pred_d = dict(zip(mod_payload, mod_preds))
        for p in problems:
            score_d = {}
            for test_comp in self.test_fns:
                t1, t2 = test_comp.build_pair_fn(p)
                key = t1, t2
                if key in mod_mapping:
                    new_key = mod_mapping[key]
                    pred = mod_pred_d[new_key]
                else:
                    pred = preds_d[key]

                score_d[test_comp.name] = test_comp.score_getter(pred)

            combined_score = self.score_combine(score_d)
            out_scores.append(combined_score)
        return out_scores

    def get_not_important_indices(self, tli, t1, t2):
        tokens = t2.split()
        neutral_indices = get_neutral_reasons(tli)
        output = []
        for i in neutral_indices:
            token = tokens[i]
            if not self.is_important_fn(token):
                output.append(i)
        return output


def remove_indices(s, indices):
    tokens = s.split()
    output = []
    for i, token in enumerate(tokens):
        if i in indices:
            output.append(token)
    return " ".join(output)


def get_post_fix_idf_ignore(split) -> PostFixModel:
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
    def combine(score_d):
        opt1 = score_d["entail(t1, q+ys)"] + score_d["entail(t2, q+no)"]
        opt2 = score_d["entail(t1, q+no)"] + score_d["entail(t2, q+ys)"]
        opt3 = score_d["cont(t1, q+no)"] + score_d["cont(t2, q+ys)"]
        opt4 = score_d["cont(t1, q+ys)"] + score_d["cont(t2, q+no)"]
        condition1 = max(opt1, opt2, opt3, opt4)
        condition2 = score_d["cont(t1, t2)"]
        return (condition1 + condition2) / 2

    nli_predict_fn = get_pep_cache_client()
    question, claims = get_bioclaim_retrieval_corpus(split)
    df, N, avdl = build_stats(right(question))
    def term_idf_factor(df):
        return math.log((N - df + 0.5) / (df + 0.5))

    tokenizer = KrovetzNLTKTokenizer()
    def is_important_fn(token):
        idf_list = []
        for t in tokenizer.tokenize_stem(token):
            idf = term_idf_factor(df[t])
            idf_list.append(idf)
        idf = max(idf_list)
        return idf > 1

    return PostFixModel(nli_predict_fn, enum_subseq_136, test_fns, combine, is_important_fn)