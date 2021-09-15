from typing import List, Dict

from contradiction.medical_claims.annotation_1.mturk_scheme import PairedIndicesLabel, AlamriLabelUnit
from list_lib import flatten, lmap
from misc_lib import group_by, get_first
from trec.trec_parse import write_trec_relevance_judgement
from trec.types import TrecRelevanceJudgementEntry


def combine_alamri1_annots(annots, combine_method) -> List[AlamriLabelUnit]:
    grouped = group_by(annots, get_first)


    selected_annots = []
    for hit_id, entries in grouped.items():
        final_e = combine_method(entries)
        selected_annots.append(final_e)
    return selected_annots


def convert_annots_to_json_serializable(annots: List[AlamriLabelUnit]):
    def to_dict(e: AlamriLabelUnit) -> Dict:
        (group_no, idx), annot = e
        return {
            'group_no': group_no,
            'inner_idx': idx,
            'label': annot.to_dict()
        }

    out = list(map(to_dict, annots))
    return out


def json_dict_list_to_annots(maybe_list: List) -> List[AlamriLabelUnit]:
    def convert(d: Dict) -> AlamriLabelUnit:
        label = d['label']
        return (d['group_no'], d['inner_idx']), PairedIndicesLabel.from_dict(label)

    return lmap(convert, maybe_list)


def label_to_trec_entries(e: AlamriLabelUnit)\
        -> List[TrecRelevanceJudgementEntry]:
    data_id, label = e
    group_no, inner_no = data_id
    d = label.to_dict()
    output = []
    for sent_type, indices in d.items():
        query_id = "{}_{}_{}".format(group_no, inner_no, sent_type)
        for idx in indices:
            doc_id = str(idx)
            judge = TrecRelevanceJudgementEntry(query_id, doc_id, 1)
            output.append(judge)
    return output


def save_annots_to_qrel(annots: List[AlamriLabelUnit],
                        save_path: str):
    rel_entries = flatten(map(label_to_trec_entries, annots))
    write_trec_relevance_judgement(rel_entries, save_path)