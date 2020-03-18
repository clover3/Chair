import string

import math
import nltk

from arg.claim_building.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.perspectives.basic_analysis import get_candidates
from arg.perspectives.collection_based_classifier import CollectionInterface, re_tokenize
from arg.perspectives.load import get_claims_from_ids, load_train_claim_ids
from list_lib import lmap, idx_where, lfilter
from visualize.html_visual import HtmlVisualizer, Cell


def binary_feature_demo(datapoint_list):
    ci = CollectionInterface()
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


    print_cnt = 0
    for dp_idx, data_point in enumerate(datapoint_list):
        label, cid, pid, claim_text, p_text = data_point
        query_id = "{}_{}".format(cid, pid)
        ranked_docs = ci.get_ranked_documents_tf(cid, pid, True)
        ranked_list = ci.get_ranked_list(query_id)
        html.write_paragraph(claim_text)
        html.write_paragraph(p_text)
        html.write_paragraph("{}".format(label))

        local_print_cnt = 0
        lines = []
        for ranked_entry, doc in zip(ranked_list, ranked_docs):
            doc_id, orig_rank, galago_score = ranked_entry
            if doc is not None:
                score, max_score, mentioned_terms = idf_scorer(doc, claim_text, p_text)
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

                #html.write_paragraph("{0} / {1:.2f}".format(doc_id, galago_score))
                #html.write_paragraph("{}/{}".format(score, max_score))
                #html.multirow_print(lmap(get_cell, tokens))
                # local_print_cnt += 1
                # if local_print_cnt > 10:
                #     break

        matched_idx = idx_where(lambda x: x[2], lines)
        if not matched_idx:
            html.write_paragraph("No match")
        else:
            last_matched = matched_idx[-1]
            lines = lines[:last_matched+1]
            rows = lmap(lambda line: lmap(Cell, line), lines)
            html.write_table(rows)

        if dp_idx > 50:
            break


def work():
    d_ids = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    all_data_points = get_candidates(claims)
    binary_feature_demo(all_data_points)


if __name__ == "__main__":
    work()
