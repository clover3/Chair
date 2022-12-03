from typing import List, Tuple, Dict

from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from dataset_specific.ists.parse import iSTSProblem, AlignmentPredictionList, simplify_type, ALIGN_EQUI, ALIGN_SPE1, \
    ALIGN_SPE2, ALIGN_NOALI, ALIGN_SIMI, ALIGN_REL, ALIGN_OPPO
from list_lib import index_by_fn
from trec.types import TrecRelevanceJudgementEntry


def build_is_entailed(aligns, tokens1, tokens2) -> Tuple[List[bool], List[bool]]:
    is_entailed1: Dict[int, bool] = {}
    is_entailed2: Dict[int, bool] = {}

    def set_values(is_entailed: Dict[int, bool], ids: List[int], v: bool):
        for i in ids:
            is_entailed[i] = v

    def set_by_exist_right(spe_side_mark, spe_side_ids, spe_side_tokens, less_spe_side_tokens):
        entailed_indices = []
        not_entailed_indices = []
        for i in spe_side_ids:
            if spe_side_tokens[i - 1] in less_spe_side_tokens:
                entailed_indices.append(i)
            else:
                not_entailed_indices.append(i)
        set_values(spe_side_mark, entailed_indices, True)
        set_values(spe_side_mark, not_entailed_indices, False)
    for a in aligns:
        base_align_type = simplify_type(a.align_types)  # remove POLI, FACT stuff
        if base_align_type == ALIGN_EQUI:  # Both are all entailed
            set_values(is_entailed1, a.chunk_token_id1, True)
            set_values(is_entailed2, a.chunk_token_id2, True)
        elif base_align_type in [ALIGN_SPE1, ALIGN_SPE2, ALIGN_NOALI]:  # One is not all entailed
            if base_align_type == ALIGN_SPE1:
                more_specific_sent_idx = 0
            elif base_align_type == ALIGN_SPE2:
                more_specific_sent_idx = 1
            elif base_align_type == ALIGN_NOALI:
                def is_empty(ids):
                    return len(ids) == 1 and ids[0] == 0

                if is_empty(a.chunk_token_id1) and not is_empty(a.chunk_token_id2):
                    more_specific_sent_idx = 1
                elif not is_empty(a.chunk_token_id1) and is_empty(a.chunk_token_id2):
                    more_specific_sent_idx = 0
                else:
                    assert False
            else:
                assert False

            if more_specific_sent_idx == 0:
                spe_side_ids = a.chunk_token_id1
                spe_side_tokens = tokens1
                less_spe_side_ids = a.chunk_token_id2
                less_spe_side_tokens = tokens2
                spe_sent_idx = 0
                less_spe_sent_idx = 1
            elif more_specific_sent_idx == 1:
                spe_side_ids = a.chunk_token_id2
                spe_side_tokens = tokens2
                less_spe_side_ids = a.chunk_token_id1
                less_spe_side_tokens = tokens1
                spe_sent_idx = 1
                less_spe_sent_idx = 0
            else:
                assert False

            spe_side_mark = [is_entailed1, is_entailed2][spe_sent_idx]
            less_spe_side_mark = [is_entailed1, is_entailed2][less_spe_sent_idx]
            set_by_exist_right(spe_side_mark, spe_side_ids, spe_side_tokens, less_spe_side_tokens)
            set_values(less_spe_side_mark, less_spe_side_ids, True)
        elif base_align_type in [ALIGN_SIMI, ALIGN_REL, ALIGN_OPPO, ALIGN_EQUI]:
            set_by_exist_right(is_entailed1, a.chunk_token_id1, tokens1, tokens2)
            set_by_exist_right(is_entailed2, a.chunk_token_id2, tokens2, tokens1)
        else:
            print(base_align_type)
            assert False

    left_labels: List[bool] = []
    right_labels: List[bool] = []
    for i, _ in enumerate(tokens1):
        left_labels.append(is_entailed1[i+1])

    for i, _ in enumerate(tokens2):
        right_labels.append(is_entailed2[i+1])
    return left_labels, right_labels


def convert_to_sent_token_label(problems: List[iSTSProblem], labels: AlignmentPredictionList) -> List[SentTokenLabel]:
    problem_d = index_by_fn(lambda x: x.problem_id, problems)

    output = []
    for problem_id, aligns in labels:
        p = problem_d[problem_id]
        tokens1 = p.text1.split()
        tokens2 = p.text2.split()
        try:
            label: Tuple[List[bool], List[bool]] = build_is_entailed(aligns, tokens1, tokens2)
            is_entailed1, is_entailed2 = label
        except KeyError:
            if problem_id in ["252", "287", "315"]:
                print("Skip missing label: ", problem_id)
                continue
            else:
                raise
        def to_sent_token_label(sent_no, is_entailed: List[bool]):
            not_entailed: List[bool] = [not v for v in is_entailed]
            label: List[int] = list(map(int, not_entailed))
            qid = "{}-{}".format(problem_id, sent_no)

            return SentTokenLabel(qid, label)
        s1 = to_sent_token_label(1, is_entailed1)
        output.append(s1)
        output.append(to_sent_token_label(2, is_entailed2))
    return output


def sent_token_label_to_trec_qrel(label_list: List[SentTokenLabel]) -> List[TrecRelevanceJudgementEntry]:
    def convert_one(item: SentTokenLabel):
        for i, v in enumerate(item.labels):
            yield TrecRelevanceJudgementEntry(item.qid, str(i+1), v)

    entries = []
    for item in label_list:
        entries.extend(convert_one(item))
    return entries