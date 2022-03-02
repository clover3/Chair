from typing import Tuple, List, Dict

from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText, SWTTIndex

SplittedDoc = Tuple[str, SegmentwiseTokenizedText, List[Tuple[SWTTIndex, SWTTIndex]]]


def sd_to_json(sd: SplittedDoc) -> Dict:
    doc_id, doc, window_list = sd
    window_list_json = [(st.to_json(), ed.to_json()) for st, ed in window_list]
    return {
        'doc_id': doc_id,
        'doc': doc.to_json(),
        'window_list': window_list_json,
    }


def sd_from_json(j) -> SplittedDoc:
    doc_id = j['doc_id']
    doc_j = j['doc']
    window_list_j = j['window_list']

    window_list: List[Tuple[SWTTIndex, SWTTIndex]]\
        = [(SWTTIndex.from_json(st), SWTTIndex.from_json(ed)) for st, ed in window_list_j]

    doc = SegmentwiseTokenizedText.from_json(doc_j)
    return doc_id, doc, window_list
