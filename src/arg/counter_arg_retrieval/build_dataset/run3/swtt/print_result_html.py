import os
from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.ca_query import CAQuery
from arg.counter_arg_retrieval.build_dataset.run3.highlight_selector import HighlightSelector
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from cache import load_from_pickle, load_pickle_from
from cpath import output_path
from list_lib import right, foreach
from misc_lib import exist_or_mkdir, get_dir_files
from models.classic.stemming import StemmedToken
from visualize.html_visual import HtmlVisualizer, Cell, apply_html_highlight


def read_save(html_save_dir, prediction_entries):
    DocumentRepresentation = SegmentwiseTokenizedText
    docs: List[Tuple[str, DocumentRepresentation]] = load_from_pickle("ca_run3_swtt")
    docs_d = dict(docs)
    threshold = 0.5
    duplicate_indices = DocumentRepresentation.get_duplicate(right(docs))
    duplicate_doc_ids = [docs[idx][0] for idx in duplicate_indices]
    hs = HighlightSelector()
    exist_or_mkdir(html_save_dir)
    mark_color = "mark { background-color: #7FFFFF; color: black; }"
    passage_format = ".passage { line-height: 1.8; }"
    for ca_query, docs_and_scores in prediction_entries:
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
                    st, ed = scores.windows_st_ed_list[passage_idx]
                    word_tokens_grouped: List[List[str]] = doc.get_word_tokens_grouped(st, ed)
                    text_html_all = ""
                    for word_tokens in word_tokens_grouped:
                        p_stemmed_tokens = hs.stem_tokens(word_tokens)
                        highlight_indices = hs.get_highlight_indices_inner(q_stemmed_tokens, p_stemmed_tokens)
                        highlighted_html_text = apply_html_highlight(highlight_indices, word_tokens)
                        text_html = "<div class={}>{}</div>".format("passage", highlighted_html_text)
                        text_html_all += text_html
                    rows.append(list(map(Cell, [passage_idx, text_html_all, score])))

            if len(rows) > 1:
                n_rel_doc += 1
                # print_table(rows)
                html.write_table(rows)
        print(f"{n_rel_doc} of {n_non_duplicate} docs are relevants ({n_rel_passage} passages)")


def read_save_default(run_name):
    html_save_dir = os.path.join(output_path, "ca_building", "run3", "{}_html".format(run_name))
    prediction_entries: List[Tuple[CAQuery, List[Tuple[str, SWTTScorerOutput]]]] = []

    save_dir = os.path.join(output_path, "ca_building", "run3", run_name)
    for file_path in get_dir_files(save_dir):
        prediction_entries.extend(load_pickle_from(file_path))

    read_save(html_save_dir, prediction_entries)


def main():
    read_save_default("PQ_2")


if __name__ == "__main__":
    main()

