import math
from typing import List, Tuple

from krovetzstemmer import Stemmer

from adhoc.clueweb12_B13_termstat import load_clueweb12_B13_termstat_stemmed, cdf
from contradiction.medical_claims.token_tagging.eval_helper import do_ecc_eval_w_trec_eval
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from contradiction.medical_claims.token_tagging.runner_s.run_exact_match import solve_and_save
from contradiction.medical_claims.token_tagging.tf_idf_discretize import daily_topic_rare
from misc_lib import tprint


# Assign 1 if no exact match exists
class TF_IDF_AnalysisHelper(TokenScoringSolverIF):
    def __init__(self, df, ctf, stemmer):
        self.ctf = ctf
        self.df = df
        self.stemmer = stemmer

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = self.overlap_check_1_if_no(text1_tokens, text2_tokens)
        scores2 = self.overlap_check_1_if_no(text2_tokens, text1_tokens)
        return scores1, scores2

    def get_weight(self, token) -> float:
        stemmed_token = self.stemmer(token)
        df = self.df[stemmed_token]
        # if df == 0:
        #     df = 10

        assert self.ctf - df + 0.5 > 0
        raw_weight = math.log((self.ctf - df + 0.5) / (df + 0.5))
        return daily_topic_rare(raw_weight)

    def overlap_check_1_if_no(self, text1_tokens, text2_tokens) -> List[float]:
        scores = []
        for t1 in text1_tokens:
            weight = self.get_weight(t1)
            assert weight > 0
            if t1 in text2_tokens:
                tf = sum([1 if t1 == t2 else 0 for t2 in text2_tokens])
                assert tf > 0
                match_score = -tf
            else:
                match_score = 1


            if match_score < 0: # match exists
                s = match_score * weight
            else: # no match
                assert match_score == 1
                s = match_score * weight
            scores.append(s)
        return scores


def run_tf_idf():
    tprint("Loading term stats")
    tf, df = load_clueweb12_B13_termstat_stemmed()
    stemmer = Stemmer()
    tprint("building solver")
    tag_type = "mismatch"
    run_name = "tf_idf_analysis"
    solver = TF_IDF_AnalysisHelper(df, cdf, stemmer)
    tprint("running solver")
    solve_and_save(run_name, solver, tag_type)
    do_ecc_eval_w_trec_eval(run_name, tag_type)



if __name__ == "__main__":
    run_tf_idf()
