from typing import List

from contradiction.medical_claims.annotation_1.mturk_scheme import load_all_annotation_w_reject, \
    load_all_annotation, parse_alamri_hit, MTurkOutputFormatError, parse_hit_with_indices_fix
from contradiction.medical_claims.annotation_1.worker_id_info import get_worker_list_to_reject, workers_to_filter, \
    trusted_worker
from contradiction.medical_claims.label_structure import AlamriLabelUnitT
from mturk.parse_util import HitResult


def load_annots_w_processing() -> List[AlamriLabelUnitT]:
    exclude_worker_ids = get_worker_list_to_reject()
    exclude_worker_ids.extend(workers_to_filter)
    hits_inc_reject = load_all_annotation_w_reject()
    print(f"{len(hits_inc_reject)} all including rejected hits")
    hits: List[HitResult] = load_all_annotation()
    print(f"{len(hits)} all non-rejected hits")
    hits_filtered = list(filter(lambda h: h.worker_id not in exclude_worker_ids, hits))
    print(f"{len(hits_filtered)} from non-black list")
    invalid = []

    def parse_hit(h) -> AlamriLabelUnitT:
        try:
            return parse_alamri_hit(h)
        except MTurkOutputFormatError:
            invalid.append(h)
            print(h.worker_id, h.outputs['indices'])
            return None

    annots: List[AlamriLabelUnitT] = list(filter(None, map(parse_hit, hits_filtered)))
    print(f"{len(annots)} acquired")
    print("{} are broken".format(len(invalid)))
    trusted_worker = ['A1J1MXAI07HGUT', 'A1QE4E0WPJZGEI']
    added_annot = []
    for hit in invalid:
        if hit.worker_id in trusted_worker:
            annot = parse_hit_with_indices_fix(hit)
            added_annot.append(annot)
    print(f"add {len(added_annot)} annots by fixing")
    all_annots: List[AlamriLabelUnitT] = annots + added_annot
    return all_annots


def load_annots_for_worker(target_worker) -> List[AlamriLabelUnitT]:
    exclude_worker_ids = get_worker_list_to_reject()
    exclude_worker_ids.extend(workers_to_filter)
    hits_inc_reject = load_all_annotation_w_reject()
    print(f"{len(hits_inc_reject)} all including rejected hits")
    hits: List[HitResult] = load_all_annotation()
    print(f"{len(hits)} all non-rejected hits")
    hits_filtered = list(filter(lambda h: h.worker_id == target_worker, hits))
    print(f"{len(hits_filtered)} from non-black list")
    invalid = []
    def parse_hit(h) -> AlamriLabelUnitT:
        try:
            return parse_alamri_hit(h)
        except MTurkOutputFormatError:
            invalid.append(h)
            print(h.worker_id, h.outputs['indices'])
            return None

    annots: List[AlamriLabelUnitT] = list(filter(None, map(parse_hit, hits_filtered)))
    print(f"{len(annots)} acquired")
    print("{} are broken".format(len(invalid)))
    added_annot = []
    for hit in invalid:
        if hit.worker_id in trusted_worker:
            annot = parse_hit_with_indices_fix(hit)
            added_annot.append(annot)
    print(f"add {len(added_annot)} annots by fixing")
    all_annots: List[AlamriLabelUnitT] = annots + added_annot
    return all_annots

