import pickle
from typing import List

import nltk

from arg.perspectives.declaration import ParagraphClaimPersFeature
from base_type import FileName
from cpath import pjoin, output_path
from visualize.html_visual import HtmlVisualizer, Cell


def show(html_visualizer: HtmlVisualizer, features: List[ParagraphClaimPersFeature]):
    print("Cid: ", features[0].claim_pers.cid)
    for f in features:
        html_visualizer.write_paragraph("Claim: " + f.claim_pers.claim_text)
        html_visualizer.write_paragraph("Perspective: " + f.claim_pers.p_text)

        pc_tokens: List[str] = nltk.word_tokenize(f.claim_pers.claim_text) + nltk.word_tokenize(f.claim_pers.p_text)
        pc_tokens_set = set([t.lower() for t in pc_tokens])
        print(pc_tokens_set)
        def get_cell(token) -> Cell:
            if token.lower() in pc_tokens_set:
                score = 100
            else:
                score = 0
            return Cell(token, score)

        html_visualizer.write_paragraph("Label : {}".format(f.claim_pers.label))
        for score_paragraph in f.feature:
            paragraph = score_paragraph.paragraph
            cells = [get_cell(t) for t in paragraph.tokens]
            html_visualizer.write_paragraph("---")
            html_visualizer.multirow_print(cells, width=20)


if __name__ == "__main__":
    input_job_name: FileName = FileName("perspective_paragraph_feature_dev")
    input_dir = pjoin(output_path, input_job_name)
    job_id = 0
    features: List[ParagraphClaimPersFeature] = pickle.load(open(pjoin(input_dir, FileName(str(job_id))), "rb"))
    html = HtmlVisualizer("pers_dev_para_features.html")
    show(html, features)

