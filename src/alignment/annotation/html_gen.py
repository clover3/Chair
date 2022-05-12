import os

from contradiction.medical_claims.annotation_1.annotation_html_gen import escape_sentence
from cpath import src_path
from dataset_specific.mnli.mnli_reader import NLIPairData


def get_get_html_fn():
    html_template_path = os.path.join(src_path, "html", "token_annotation", "align_annotation_template.html")
    html_template = open(html_template_path, "r").read()

    def get_html(nli_pair: NLIPairData) -> str:
        replace_mapping = {
            'SuperWildCardMyIfThePremise': escape_sentence(nli_pair.premise),
            'SuperWildCardMyIfTheHypothesis': escape_sentence(nli_pair.hypothesis),
            "SuperWildCardMyIfID": str(nli_pair.data_id)
        }
        new_html = html_template
        for place_holder, value in replace_mapping.items():
            new_html = new_html.replace(place_holder, value)
        return new_html
    return get_html