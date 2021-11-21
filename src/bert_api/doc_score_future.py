
from typing import List, Tuple

from bert_api.client_lib_msmarco import BERTClientMSMarco
from bert_api.doc_score_defs import DocumentScorerOutputSbword, DocumentScorerInput
from data_generator.tokenize_helper import TokenizedText, SbwordIdx
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import right
from misc_lib import split_window
from trainer.promise import PromiseKeeper, MyPromise, MyFuture


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
