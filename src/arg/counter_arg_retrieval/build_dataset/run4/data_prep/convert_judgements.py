from typing import Tuple, Dict, List, NamedTuple

from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from arg.counter_arg_retrieval.build_dataset.path_helper import load_sliced_passage_ranked_list
from arg.counter_arg_retrieval.build_dataset.run4.run4_util import load_run4_swtt_passage
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText, IntTuple
from trec.types import DocID


class Judgement(NamedTuple):
    qid: str
    doc_id: str
    passage_st: IntTuple
    passage_ed: IntTuple

    def __eq__(self, other):
        return self.qid == other.qid and \
               self.doc_id == other.doc_id and \
               self.passage_st == other.passage_st and \
               self.passage_ed == other.passage_ed


def document_overlap():
    run_name = "PQ_1"
    all_doc_ids = get_doc_ids_from_run(run_name)
    print(len(all_doc_ids))

    runs = ["PQ_6", "PQ_7", "PQ_8", "PQ_9"]
    for run_name in runs:
        doc_ids = get_doc_ids_from_run(run_name)
        n_common = len(all_doc_ids.intersection(doc_ids))
        print(n_common, len(doc_ids), run_name)




def get_doc_ids_from_run(run_name):
    pq1 = load_sliced_passage_ranked_list(run_name)
    all_doc_ids = set()
    for qid, entries in pq1.items():
        passage_doc_ids = [e.doc_id for e in entries]
        doc_id_list = []
        for p_doc_id in passage_doc_ids:
            doc_id, passage_idx = p_doc_id.split("_")
            doc_id_list.append(doc_id)

        all_doc_ids.update(doc_id_list)
    return all_doc_ids


def main():
    passages: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]]\
        = load_run4_swtt_passage()
    run_name = "PQ_1"
    pq1 = load_sliced_passage_ranked_list(run_name)
    all_judgements = convert_to_judgment_entries(passages, pq1)

    runs = ["PQ_6", "PQ_7", "PQ_8", "PQ_9"]
    for run_name in runs:
        pq = load_sliced_passage_ranked_list(run_name)
        required_judgements = convert_to_judgment_entries(passages, pq)
        print(run_name)
        for e in required_judgements:
            if e in all_judgements:
                print(e)



def convert_to_judgment_entries(passages, pq1):
    all_judgements = []
    for qid, entries in pq1.items():
        for e in entries:
            doc_id, passage_idx = e.doc_id.split("_")
            swtt, passage_ranges = passages[doc_id]
            st, ed = passage_ranges[passage_idx]
            judgement = Judgement(qid, doc_id, st, ed)
            all_judgements.append(judgement)
    return all_judgements


if __name__ == "__main__":
    main()