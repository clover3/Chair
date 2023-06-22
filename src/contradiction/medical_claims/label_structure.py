from typing import NamedTuple, List, Tuple


class PairedIndicesLabel(NamedTuple):
    prem_conflict_indices: List[int]
    prem_mismatch_indices: List[int]
    hypo_conflict_indices: List[int]
    hypo_mismatch_indices: List[int]

    def enum(self):
        yield self.prem_conflict_indices
        yield self.prem_mismatch_indices
        yield self.hypo_conflict_indices
        yield self.hypo_mismatch_indices

    def to_dict(self):
        return {
            'prem_conflict': self.prem_conflict_indices,
            'prem_mismatch': self.prem_mismatch_indices,
            'hypo_conflict': self.hypo_conflict_indices,
            'hypo_mismatch': self.hypo_mismatch_indices,
        }

    @staticmethod
    def from_dict(d):
        return PairedIndicesLabel(
            d['prem_conflict'],
            d['prem_mismatch'],
            d['hypo_conflict'],
            d['hypo_mismatch'],
        )

    @staticmethod
    def get_sent_types():
        return [
            "prem_conflict",
            "prem_mismatch",
            'hypo_conflict',
            'hypo_mismatch'
        ]

    def get_label_tuple(self, tag) -> Tuple[List[bool], List[bool]]:
        maybe_seq_len = max(map(len, self.to_dict().values())) + 1
        def to_binary(indices) -> List[bool]:
            return [i in indices for i in range(maybe_seq_len)]

        if tag == "conflict":
            return to_binary(self.prem_conflict_indices), to_binary(self.hypo_conflict_indices)
        elif tag == "mismatch":
            return to_binary(self.prem_mismatch_indices), to_binary(self.hypo_mismatch_indices)
        else:
            raise ValueError


AlamriLabelUnitT = Tuple[Tuple[int, int], PairedIndicesLabel]


class AlamriLabel(NamedTuple):
    group_no: int
    inner_idx: int
    label: PairedIndicesLabel

    @classmethod
    def from_tuple(cls, tuple: AlamriLabelUnitT):
        (group_no, inner_idx), label = tuple
        return AlamriLabel(group_no, inner_idx, label)
