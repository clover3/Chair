import os
from typing import NamedTuple, List

# SR :SegmentResource
from cache import load_pickle_from


class SegmentRepresentation(NamedTuple):
    first_seg: List[int]
    second_seg: List[int]
    #
    # def serialize(self) -> bytearray:
    #     bytes_seg1 = serialize_int_list(self.first_seg)
    #     b1_l = len(bytes_seg1)
    #     bytes_seg2 = serialize_int_list(self.second_seg)
    #     b2_l = len(bytes_seg2)
    #     return to_big_bytes(b1_l) + bytes_seg1 + to_big_bytes(b2_l) + bytes_seg2
    #
    # @classmethod
    # def deserialize(cls, ba: bytearray):
    #     b1_l = from_big_bytes(ba[:2])
    #     bytes_seg1 = ba[2:2+b1_l]
    #     b2_l = from_big_bytes(ba[2+b1_l:4+b1_l])
    #     bytes_seg2 = ba[4+b1_l:]
    #     assert b2_l == len(ba) - (4+b1_l)
    #     first_seg: List[int] = deserialize_int_list(bytes_seg1)
    #     second_seg: List[int] = deserialize_int_list(bytes_seg2)
    #     return SegmentRepresentation(first_seg, second_seg)


class SRPerQueryDoc(NamedTuple):
    doc_id: str
    segs: List[SegmentRepresentation]
    label: int
    def get_label(self):
        return self.label
    #
    # def serialize(self) -> bytearray:
    #     all_bytes = bytearray()
    #     doc_id_bytes = to_utf8_bytes(self.doc_id)
    #     doc_id_len = len(doc_id_bytes)
    #     doc_id_len_bytes = to_big_bytes(doc_id_len)
    #     all_bytes.extend(doc_id_len_bytes)
    #     all_bytes.extend(doc_id_bytes)
    #     label_bytes = to_big_bytes(self.label)
    #     all_bytes.extend(label_bytes)
    #
    #     segs_bytes_list = list([s.serialize() for s in self.segs])
    #     for sb in segs_bytes_list:
    #         sb_len_bytes = len(sb)
    #         all_bytes.extend(sb)
    #
    #
    #
    # @classmethod
    # def deserialize(cls, ba: bytearray):
    #     return NotImplemented


class SRPerQuery(NamedTuple):
    qid: str
    sr_per_query_doc: List[SRPerQueryDoc]
    #
    # def serialize(self):
    #     return NotImplemented


class SegmentResourceLoader:
    def __init__(self, root_dir, split):
        self.root_dir = os.path.join(root_dir, "seg_resource_{}".format(split))

    def load_for_qid(self, qid) -> SRPerQuery:
        return load_pickle_from(os.path.join(self.root_dir, qid))

