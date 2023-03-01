from typing import List, Iterable, Callable, Dict, Tuple, Set

import nltk

from adhoc.bm25_ex import StemmerCache
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from dataset_specific.msmarco.common import load_queries, MSMarcoDoc, load_per_query_docs
from list_lib import lmap, index_by_fn
from trec.qrel_parse import load_qrels_flat_per_query
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry
import sys
from trec.types import TrecRankedListEntry
from cpath import output_path, data_path
from misc_lib import path_join, average, pause_hook, find_max_idx
from visualize.html_visual import HtmlVisualizer, Cell
import numpy as np


def get_snippet_generator():
    tokenizer = KrovetzNLTKTokenizer()
    stemmer = tokenizer.stemmer
    n_gram_n = 40
    def generate_snippets(query, title, body) -> str:
        # print n-gram with highest tf scores
        q_terms = tokenizer.tokenize_stem(query)
        n_show = 2

        def do_for_text(text, n_show):
            tokens = nltk.tokenize.word_tokenize(text)
            stemmed_tokens = [stemmer.stem(t) for t in tokens]

            st_list = range(len(tokens))
            def get_score_by_st(st):
                sub_tokens = tokens[st:st+n_gram_n]
                stemmed_sub_tokens = stemmed_tokens[st:st+n_gram_n]
                try:

                    n_match = 0
                    for q_term in q_terms:
                        if q_term in stemmed_sub_tokens:
                            n_match += 1
                except UnicodeEncodeError as e:
                    print(e)
                    return 0
                return n_match

            scores = list(map(get_score_by_st, st_list))
            rank = np.argsort(scores)[::-1]
            out_msgs = []
            for i in rank:
                st = st_list[i]
                sub_tokens = tokens[st:st + n_gram_n]
                stemmed_sub_tokens = stemmed_tokens[st:st+n_gram_n]
                skip = False
                for prev_rank in rank[:i]:
                    st_prev = st_list[prev_rank]
                    dist = abs(st_prev - st)
                    if dist < 0.5 * n_gram_n:
                        skip = True
                if skip:
                    continue
                try:

                    matched_indices = []
                    for q_term in q_terms:
                        if q_term in stemmed_sub_tokens:
                            matched_indices.append(stemmed_sub_tokens.index(q_term))

                    display_tokens = []
                    for i, t in enumerate(sub_tokens):
                        if stemmed_sub_tokens[i] in q_terms:
                            display_tokens.append("<b>{}</b>".format(t))
                        else:
                            display_tokens.append(t)

                except UnicodeEncodeError as e:
                    print(e)
                    return 0
                out_msgs.append(" ".join(display_tokens))

                if len(out_msgs) >= n_show:
                    break
            return out_msgs

        msg_list = do_for_text(title, 1) + do_for_text(body, n_show)
        return "<br>".join(msg_list)

    return generate_snippets






def main():
    qrel_path = path_join(data_path, "msmarco",  "msmarco-docdev-qrels.tsv")
    ranked_list_path = path_join(output_path, "ranked_list",  "mmd_Z_50000_dev_a_small_reduced.txt")
    queries_d = dict(load_queries("dev"))
    ranked_list = load_ranked_list_grouped(ranked_list_path)
    qrels = load_qrels_flat_per_query(qrel_path)
    not_found = 0

    generate_snippets = get_snippet_generator()
    save_path = "msmarco_mmd_z_result.html"
    html = HtmlVisualizer(save_path)
    rr_list = []
    out_cnt = 0
    for query_id in ranked_list:
        q_ranked_list: List[TrecRankedListEntry] = ranked_list[query_id]
        query = queries_d[query_id]
        try:
            gold_list = qrels[query_id]
            true_gold: List[str] = list([doc_id for doc_id, score in gold_list if score > 0])

            true_rank = []
            for e in q_ranked_list:
                if e.doc_id in true_gold:
                    rank = e.rank + 1  # 1 to be top
                    true_rank.append(rank)

            if len(true_rank) == 1:
                rank_msg = true_rank[0]
                rr = 1 / true_rank[0]
            elif len(true_rank) == 0:
                rank_msg = "inf (not found)"
                rr = 0
            else:
                rr = 1 / true_rank[0]
                rank_msg = true_rank

            rr_list.append(rr)


            msg = "QID={} Rank={}: gold={}, {}".format(query_id, rank_msg, true_gold, query)
            print(msg)
            if true_rank and true_rank[0] > 10:
                docs: List[MSMarcoDoc] = load_per_query_docs(query_id, None)
                docs_d = index_by_fn(lambda x:x.doc_id, docs)

                head = ["doc_id", "rank", "score", "url", "is_gold"]
                table_body = []
                for e in q_ranked_list:
                    doc = docs_d[e.doc_id]
                    doc_url_link = "<a href=\"{}\">{}</a>".format(doc.url, e.doc_id)
                    score_str = "{0:.3f}".format(e.score)

                    doc_desc = generate_snippets(query, doc.title, doc.body)
                    row = [doc_url_link, e.rank+1, score_str,
                           doc_desc,
                           e.doc_id in true_gold]
                    table_body.append(lmap(Cell, row))

                html.write_paragraph(msg)
                html.write_table(table_body, head)
                html.write_bar()
                out_cnt += 1
                if out_cnt > 100:
                    break
        except KeyError as e:
            not_found += 1
    if not_found:
        print("{} of {} queires not found".format(not_found, len(ranked_list)))

    avg_rr = average(rr_list)
    print("avg rr : {}".format(avg_rr))
    return NotImplemented


if __name__ == "__main__":
    main()