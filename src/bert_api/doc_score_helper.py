from typing import List, Tuple, NamedTuple, Dict, Callable

from bert_api.client_lib_msmarco import BERTClientMSMarco
from data_generator.tokenize_helper import TokenizedText, SbwordIdx, WordIdx
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import right
from misc_lib import split_window
from trainer.promise import PromiseKeeper, MyPromise, MyFuture


class DocumentScorerOutputSbword(NamedTuple):
    window_start_loc: List[SbwordIdx]  # In sbword
    scores: List[float]


class DocumentScorerOutput(NamedTuple):
    window_start_loc: List[WordIdx]  # In sbword
    scores: List[float]

    @classmethod
    def from_dsos(cls, document_scorer_output_subword: DocumentScorerOutputSbword, doc: TokenizedText):
        window_start_loc: List[WordIdx] = []
        for sbword_idx in document_scorer_output_subword.window_start_loc:
            word_idx = WordIdx(doc.sbword_mapping[sbword_idx])
            window_start_loc.append(word_idx)
        return DocumentScorerOutput(window_start_loc, document_scorer_output_subword.scores)


class DocumentScorerInput(NamedTuple):
    window_start_loc: List[SbwordIdx]
    payload_list: List[Tuple[List[str], List[str]]]


class DocumentScorer:
    def __init__(self, client: BERTClientMSMarco, max_seg_per_document=None):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()
        self.pk: PromiseKeeper = PromiseKeeper(self.do_duty)
        self.client: BERTClientMSMarco = client
        self.max_seg_per_document = max_seg_per_document

    def do_duty(self, l: List[DocumentScorerInput]) -> List[DocumentScorerOutputSbword]:
        output_list = []
        for e in l:
            scores: List[Tuple[float, float]] = self.client.request_multiple_from_tokens(e.payload_list)
            output = DocumentScorerOutputSbword(e.window_start_loc, right(scores))
            output_list.append(output)
        return output_list

    def score_relevance(self, query_text, doc: TokenizedText) -> MyFuture:  # DocumentScorerOutputSbword:
        q_tokens = self.tokenizer.tokenize(query_text)
        available_doc_len = self.max_seq_length - len(q_tokens) - 3
        segment_list: List[List[str]] = split_window(doc.sbword_tokens, available_doc_len)

        window_start_loc: List[SbwordIdx] = []
        loc_in_sbword = 0

        payload_list = []
        if self.max_seg_per_document is not None:
            segment_list = segment_list[:self.max_seg_per_document]
        for seg_idx, segment in enumerate(segment_list):
            window_start_loc.append(SbwordIdx(loc_in_sbword))
            loc_in_sbword += len(segment)

            e = (q_tokens, segment)
            payload_list.append(e)

        promise_input = DocumentScorerInput(window_start_loc, payload_list)
        promise = MyPromise(promise_input, self.pk)
        return promise.future()


def get_cache_doc_tokenizer(docs_d: Dict[str, str]) -> Callable:
    doc_payload_d = {}
    def get_tokenized_doc(doc_id) -> TokenizedText:
        if doc_id in doc_payload_d:
            return doc_payload_d[doc_id]
        return TokenizedText.from_text(docs_d[doc_id])
    return get_tokenized_doc