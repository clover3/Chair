import re
from typing import List, Tuple, Union

from contradiction.medical_claims.annotation_1.analyze_result.read_batch import load_file_list
from contradiction.medical_claims.label_structure import PairedIndicesLabel, AlamriLabelUnitT
from mturk.parse_util import HITScheme, Textbox, ColumnName, HitResult, parse_file


def get_alamri_scheme():
    inputs = [ColumnName("url")]
    answer_units = [Textbox("indices")]
    return HITScheme(inputs, answer_units)


def load_all_annotation() -> List[HitResult]:
    scheme = get_alamri_scheme()
    results = []
    for file_path in load_file_list():
        print(file_path)
        hit_results: List[HitResult] = parse_file(file_path, scheme)
        results.extend(hit_results)
    return results


def load_all_annotation_w_reject() -> List[HitResult]:
    scheme = get_alamri_scheme()
    results = []
    for file_path in load_file_list():
        print(file_path)
        hit_results: List[HitResult] = parse_file(file_path, scheme, False)
        results.extend(hit_results)
    return results


class MTurkOutputFormatError(ValueError):
    pass


def parse_url(url_text: str) -> Tuple[int, int]:
    # url_text example: https://ecc.neocities.org/2/1.html
    prefix = "https://ecc.neocities.org/"

    if not url_text.startswith(prefix):
        raise ValueError

    tail = url_text[len(prefix):]

    pattern_text = r"(\d+)/(\d+)\.html"
    pattern = re.compile(pattern_text)
    ret = pattern.match(tail)
    if ret is None:
        raise ValueError
    group_no = int(ret.group(1))
    sub_no = int(ret.group(2))
    return group_no, sub_no


def parse_sub_indices(s):
    l = filter(None, s.strip().split(" "))
    try:
        output = list(map(int, l))
        return output
    except:
        raise


def parse_indices(indices_str: str) -> Union[PairedIndicesLabel, str]:
    try:
        segs = indices_str.split(",")
        if len(segs) != 4:
            raise ValueError
        prem_conflict, prem_mismatch, hypo_conflict, hypo_mismatch = map(parse_sub_indices, segs)
        return PairedIndicesLabel(prem_conflict, prem_mismatch, hypo_conflict, hypo_mismatch)
    except ValueError:
        raise MTurkOutputFormatError


def parse_alamri_hit(h: HitResult) -> AlamriLabelUnitT:
    group_no, inst_no = parse_url(h.inputs['url'])
    annot: PairedIndicesLabel = parse_indices(h.outputs['indices'])
    inst = (group_no, inst_no), annot
    return inst


def parse_hit_with_indices_fix(h):
    indices_str = h.outputs['indices']
    segs = indices_str.split(",")
    if len(segs) != 3:
        raise ValueError

    fixed_indices_str = "," + indices_str
    group_no, inst_no = parse_url(h.inputs['url'])
    print("Fix ", group_no, inst_no)
    annot: PairedIndicesLabel = parse_indices(fixed_indices_str)
    inst = (group_no, inst_no), annot
    return inst