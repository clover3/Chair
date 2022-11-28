from typing import List

import spacy

from alignment.lexical_alignment.runner.show_alignment_pred import get_extract_edges_top3
from bert_api import SegmentedInstance
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import transpose
from ptorch.ists.ists_predictor import get_ists_predictor
from trainer_v2.epr.spacy_segmentation import spacy_segment
from visualize.html_visual import Cell, HtmlVisualizer


def print_html(seg_inst: SegmentedInstance, alignment: List[List[float]]):
    html = HtmlVisualizer("ists_align.html")
    extract_edges = get_extract_edges_top3()
    edges_from_seg1, edges_from_seg2 = extract_edges(alignment)
    tokenizer = get_tokenizer()

    todo = [
        (seg_inst.text1, seg_inst.text2, "hypothesis", edges_from_seg1),
        (seg_inst.text2, seg_inst.text1, "premise", edges_from_seg2),
    ]

    display_max = 3
    for seg_text, other_seg_text, seg_name, edges in todo:
        head = ['Idx', "Word", "Exact match"]
        head.extend(["Edge{}".format(i) for i in range(display_max)])
        head_cells = [Cell(h) for h in head]
        table_tr: List[List] = [head_cells]
        other_seg_words = [other_seg_text.get_seg_text(tokenizer, j) for j in other_seg_text.enum_seg_idx()]
        for seg_idx in seg_text.enum_seg_idx():
            edges_from_here = edges[seg_idx]
            plain_seg_text = seg_text.get_seg_text(tokenizer, seg_idx)
            if plain_seg_text in other_seg_words:
                exact_match = "Yes"
            else:
                exact_match = "No"

            row = [str(seg_idx),
                   plain_seg_text,
                   exact_match,
                   ]
            warning = len(edges_from_here) == 0 and seg_name == "hypothesis"
            highlight_score = 50 if warning else 0
            for i in range(display_max):
                if i < len(edges_from_here):
                    target_idx, score = edges_from_here[i]
                    text = other_seg_text.get_seg_text(tokenizer, target_idx)
                    row.append("{0} ({1:.1f})".format(text, score))
                else:
                    row.append("-")

            row_cells = [Cell(t, highlight_score, target_color="R") for t in row]
            table_tr.append(row_cells)
        table = transpose(table_tr)

        for row in table:
            row[0].is_head = True

        html.write_headline(seg_name.capitalize())
        head_raw = [c.s for c in table[0]]
        html.write_table(table[1:], head_raw)
        html.write_paragraph("")

    html.write_bar()


def main():
    nlp = spacy.load("en_core_web_sm")
    def sent_to_chunks(sent):
        doc = nlp(sent)
        return list(map(str, spacy_segment(doc)))

    tokenizer = get_tokenizer()

    def get_seg_text(tokens, frame_len, is_left):
        n_pad = frame_len - len(tokens)
        if not is_left:
            tokens.append("-no-align-")
        tokens = tokens + ["[PAD]"] * n_pad
        return token_list_to_segmented_text(tokenizer, tokens)

    predictor = get_ists_predictor()
    while True:
        sent1 = input("Premise: ")
        sent2 = input("Hypothesis: ")
        tokens1 = sent_to_chunks(sent1)
        tokens2 = sent_to_chunks(sent2)
        probs = predictor.predict(tokens1, tokens2)
        print("len(tokens1), len(tokens2)", len(tokens1), len(tokens2))
        print(probs.shape)

        m = len(probs)
        frame_len = m

        def tokens_w_indices(tokens):
            return " ".join([f"{i}) {token}" for i, token in enumerate(tokens)])
        print(tokens_w_indices(tokens1))
        print(tokens_w_indices(tokens2))

        si = SegmentedInstance(
            get_seg_text(tokens1, frame_len, True),
            get_seg_text(tokens2, frame_len, False)
        )
        print_html(si, probs)


if __name__ == "__main__":
    main()
