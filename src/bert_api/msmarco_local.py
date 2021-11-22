import os
from typing import List, Tuple

from bert_api.doc_score_defs import DocumentScorerInput, DocumentScorerOutputSbword, DocumentScorerInputEx, \
    DocumentScorerOutput
from bert_api.msmarco_tokenization import EncoderUnit
from bert_api.predictor import Predictor
from cpath import data_path
from data_generator.tokenize_helper import TokenizedText, SbwordIdx
from list_lib import right
from misc_lib import split_window, TEL
from trainer.promise import MyFuture, MyPromise, PromiseKeeper

DocumentScorerPromiseSpec = Tuple[DocumentScorerInput, TokenizedText]


class DocumentScorer:
    def __init__(self, predictor: Predictor, max_seg_per_document=None):
        self.max_seq_length = 512
        self.pk: PromiseKeeper = PromiseKeeper(self.do_duty)
        self.max_seg_per_document = max_seg_per_document
        self.predictor = predictor
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.encoder = EncoderUnit(self.max_seq_length, voca_path)

    def do_duty(self, l: List[DocumentScorerInputEx]) -> List[DocumentScorerOutput]:
        def encode(t) -> Tuple[List, List, List]:
            return self.encoder.encode_token_pairs(*t)

        output_list: List[DocumentScorerOutput] = []
        for e in TEL(l):
            payload_list: List[Tuple[List[str], List[str]]] = e.payload_list
            triplet_list = list(map(encode, payload_list))
            scores: List[Tuple[float, float]] = self.predictor.predict(triplet_list)
            output_sbword = DocumentScorerOutputSbword(e.window_start_loc, right(scores))
            output: DocumentScorerOutput = DocumentScorerOutput.from_dsos(output_sbword, e.doc)
            output_list.append(output)
        return output_list

    def score_relevance(self, q_tokens: List[str], doc: TokenizedText) -> MyFuture[DocumentScorerOutput]:
        available_doc_len = self.max_seq_length - len(q_tokens) - 3
        segment_list: List[List[str]] = split_window(doc.sbword_tokens, available_doc_len)

        window_start_loc: List[SbwordIdx] = []
        loc_in_sbword = 0

        payload_list: List[Tuple[List[str], List[str]]] = []
        if self.max_seg_per_document is not None:
            segment_list = segment_list[:self.max_seg_per_document]
        for seg_idx, segment in enumerate(segment_list):
            window_start_loc.append(SbwordIdx(loc_in_sbword))
            loc_in_sbword += len(segment)

            e: Tuple[List[str], List[str]] = (q_tokens, segment)
            payload_list.append(e)

        promise_input: DocumentScorerInputEx = DocumentScorerInputEx(window_start_loc, payload_list, doc)
        promise = MyPromise(promise_input, self.pk)
        return promise.future()



