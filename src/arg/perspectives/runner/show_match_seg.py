import string
from collections import Counter
from typing import List

import math
import nltk

from adhoc.bm25 import compute_K
from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.perspectives.basic_analysis import get_candidates
from arg.perspectives.clueweb_db import load_doc
from arg.perspectives.load import get_claims_from_ids, load_train_claim_ids
from arg.perspectives.ranked_list_interface import PassageRankedListInterface, Q_CONFIG_ID_BM25, make_passage_query
from arg.pf_common.text_processing import re_tokenize
from galagos.types import GalagoRankEntry
from list_lib import lmap, idx_where, lfilter
from visualize.html_visual import HtmlVisualizer, Cell


def binary_feature_demo(datapoint_list):
    ci = PassageRankedListInterface(make_passage_query, Q_CONFIG_ID_BM25)
    not_found_set = set()
    _, clue12_13_df = load_clueweb12_B13_termstat()
    cdf = 50 * 1000 * 1000
    html = HtmlVisualizer("pc_binary_feature.html")

    def idf_scorer(doc, claim_text, perspective_text):
        cp_tokens = nltk.word_tokenize(claim_text) + nltk.word_tokenize(perspective_text)
        cp_tokens = lmap(lambda x: x.lower(), cp_tokens)
        cp_tokens = set(cp_tokens)
        mentioned_terms = lfilter(lambda x: x in doc, cp_tokens)
        mentioned_terms = re_tokenize(mentioned_terms)

        def idf(term):
            if term not in clue12_13_df:
                if term in string.printable:
                    return 0
                not_found_set.add(term)

            return math.log((cdf+0.5)/(clue12_13_df[term]+0.5))

        score = sum(lmap(idf, mentioned_terms))
        max_score = sum(lmap(idf, cp_tokens))
        # print(claim_text, perspective_text)
        # print(mentioned_terms)
        # print(score, max_score)
        return score, max_score, mentioned_terms

    def bm25_estimator(doc: Counter,
                       claim_text: str,
                       perspective_text: str):
        cp_tokens = nltk.word_tokenize(claim_text) + nltk.word_tokenize(perspective_text)
        cp_tokens = lmap(lambda x: x.lower(), cp_tokens)
        k1 = 0

        def BM25_3(f, qf, df, N, dl, avdl) -> float:
            K = compute_K(dl, avdl)
            first = math.log((N - df + 0.5) / (df + 0.5))
            second = ((k1 + 1) * f) / (K + f)
            return first * second

        dl = sum(doc.values())
        info = []
        for q_term in set(cp_tokens):
            if q_term in doc:
                score = BM25_3(doc[q_term], 0, clue12_13_df[q_term], cdf, dl, 1200)
                info.append((q_term, doc[q_term], clue12_13_df[q_term], score))
        return info


    print_cnt = 0
    for dp_idx, x in enumerate(datapoint_list):
        ranked_list: List[GalagoRankEntry] = ci.query_passage(x.cid, x.pid, x.claim_text, x.p_text)
        html.write_paragraph(x.claim_text)
        html.write_paragraph(x.p_text)
        html.write_paragraph("{}".format(x.label))

        local_print_cnt = 0
        lines = []
        for ranked_entry in ranked_list:
            try:
                doc_id = ranked_entry.doc_id
                galago_score = ranked_entry.score

                tokens = load_doc(doc_id)
                doc_tf = Counter(tokens)
                if doc_tf is not None:
                    score, max_score, mentioned_terms = idf_scorer(doc_tf, x.claim_text, x.p_text)
                    matched = score > max_score * 0.75
                else:
                    matched = "Unk"
                    score = "Unk"
                    max_score = "Unk"

                def get_cell(token):
                    if token in mentioned_terms:
                        return Cell(token, highlight_score=50)
                    else:
                        return Cell(token)
                line = [doc_id, galago_score, matched, score, max_score]
                lines.append(line)
                html.write_paragraph("{0} / {1:.2f}".format(doc_id, galago_score))
                html.write_paragraph("{}/{}".format(score, max_score))
                bm25_info = bm25_estimator(doc_tf, x.claim_text, x.p_text)
                bm25_score = sum(lmap(lambda x:x[3], bm25_info))
                html.write_paragraph("bm25 re-estimate : {}".format(bm25_score))
                html.write_paragraph("{}".format(bm25_info))
                html.multirow_print(lmap(get_cell, tokens))
                local_print_cnt += 1
                if local_print_cnt > 10:
                    break
            except KeyError:
                pass

        matched_idx = idx_where(lambda x: x[2], lines)
        if not matched_idx:
            html.write_paragraph("No match")
        else:
            last_matched = matched_idx[-1]
            lines = lines[:last_matched+1]
            rows = lmap(lambda line: lmap(Cell, line), lines)
            html.write_table(rows)

        if dp_idx > 10:
            break


def work():
    d_ids = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    is_train = True
    all_data_points = get_candidates(claims, is_train)
    all_data_points = all_data_points[:10]
    binary_feature_demo(all_data_points)


if __name__ == "__main__":
    work()
