from typing import NamedTuple

from bert_api.swtt.segmentwise_tokenized_text import IntTuple, SegmentwiseTokenizedText


class Judgement(NamedTuple):
    qid: str
    doc_id: str
    passage_idx: int
    passage_st: IntTuple
    passage_ed: IntTuple

    def __eq__(self, other):
        return self.qid == other.qid and \
               self.doc_id == other.doc_id and \
               self.passage_st == other.passage_st and \
               self.passage_ed == other.passage_ed

    def get_new_doc_id(self):
        return "{}_{}".format(self.doc_id, self.passage_idx)



class JudgementEx(NamedTuple):
    qid: str
    doc_id: str
    passage_idx: int
    passage_st: IntTuple
    passage_ed: IntTuple
    swtt: SegmentwiseTokenizedText

    def __eq__(self, other):
        return self.qid == other.qid and \
               self.doc_id == other.doc_id and \
               self.passage_st == other.passage_st and \
               self.passage_ed == other.passage_ed

    def get_new_doc_id(self):
        return "{}_{}".format(self.doc_id, self.passage_idx)