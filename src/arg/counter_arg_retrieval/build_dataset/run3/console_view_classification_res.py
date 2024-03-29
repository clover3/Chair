import os
import sys
from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.ca_query import CAQuery
from arg.counter_arg_retrieval.build_dataset.run3.highlight_selector import HighlightSelector
from bert_api.doc_score_defs import DocumentScorerOutput
from cache import load_from_pickle, load_pickle_from
from data_generator.tokenize_helper import TokenizedText, WordIdx
from list_lib import right, foreach
from misc_lib import exist_or_mkdir
from models.classic.stemming import StemmedToken
from visualize.html_visual import HtmlVisualizer, Cell, apply_html_highlight


def main():
    prediction_pickle_path = sys.argv[1]
    tsv_path = sys.argv[2]
    DocumentRepresentation = TokenizedText
    docs: List[Tuple[str, DocumentRepresentation]] = load_from_pickle("ca_run3_document_processed")
    docs_d = dict(docs)
    threshold = 0.5
    output: List[Tuple[CAQuery, List[Tuple[str, DocumentScorerOutput]]]] = load_pickle_from(prediction_pickle_path)

    duplicate_indices = DocumentRepresentation.get_duplicate(right(docs))
    duplicate_doc_ids = [docs[idx][0] for idx in duplicate_indices]
    html_save_dir = prediction_pickle_path + "html"
    hs = HighlightSelector()
    exist_or_mkdir(html_save_dir)
    mark_color = "mark { background-color: #7FFFFF; color: black; }"
    passage_format = ".passage { line-height: 1.8; }"
    for ca_query, docs_and_scores in output:
        save_name = "{}_{}.html".format(ca_query.qid, ca_query.ca_query.split()[0])
        html = HtmlVisualizer(os.path.join(html_save_dir, save_name), additional_styles=[mark_color, passage_format])
        q_stemmed_tokens: List[StemmedToken] = hs.split_stem_text(ca_query.ca_query)

        n_rel_passage = 0
        n_rel_doc = 0
        s_list = [
            "Claim {}: {}".format(ca_query.qid, ca_query.claim),
            "Premise: {}".format(ca_query.perspective),
            "CA: {}".format(ca_query.ca_query),
        ]
        foreach(html.write_paragraph, s_list)
        n_non_duplicate = 0
        for doc_id, scores in docs_and_scores:
            if doc_id in duplicate_doc_ids:
                continue
            n_non_duplicate += 1
            doc = docs_d[doc_id]
            rows = [[Cell(doc_id)]]
            for passage_idx, score in enumerate(scores.scores):
                if score > threshold:
                    n_rel_passage += 1
                    st: WordIdx = scores.window_start_loc[passage_idx]
                    try:
                        ed: WordIdx = scores.window_start_loc[passage_idx+1]
                    except IndexError:
                        ed = WordIdx(-1)

                    word_tokens = doc.get_word_tokens(st, ed)

                    p_stemmed_tokens = hs.stem_tokens(word_tokens)
                    highlight_indices = hs.get_highlight_indices_inner(q_stemmed_tokens, p_stemmed_tokens)
                    highlighted_html_text = apply_html_highlight(highlight_indices, word_tokens)
                    text_html = "<div class={}>{}</div>".format("passage", highlighted_html_text)
                    rows.append(list(map(Cell, [passage_idx, text_html, score])))

            if len(rows) > 1:
                n_rel_doc += 1
                # print_table(rows)
                html.write_table(rows)
        print(f"{n_rel_doc} of {n_non_duplicate} docs are relevants ({n_rel_passage} passages)")



if __name__ == "__main__":
    main()

