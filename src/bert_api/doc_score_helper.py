from typing import List, Tuple, NamedTuple, NewType

from bert_api.client_lib import BERTClient
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import right
from misc_lib import split_window
from trainer.promise import PromiseKeeper, MyPromise, MyFuture

g_tokenizer = None


class TokenizedText(NamedTuple):
    text: str
    tokens: List[str]
    sbword_tokens: List[str]
    sbword_mapping: List[int]

    @classmethod
    def from_text(cls, text):
        global g_tokenizer
        if g_tokenizer is None:
            g_tokenizer = get_tokenizer()

        tokens = text.split()
        idx_mapping = []
        subword_list = []
        for idx, token in enumerate(tokens):
            sb_tokens = g_tokenizer.tokenize(token)
            idx_mapping.extend([idx] * len(sb_tokens))
            subword_list.extend(sb_tokens)

        return TokenizedText(text, tokens, subword_list, idx_mapping)


SbwordIdx = NewType('SubwordIndex', int)
WordIdx = NewType('WordIdx', int)


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
    def __init__(self, client: BERTClient, max_seg_per_document=None):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()
        self.pk: PromiseKeeper = PromiseKeeper(self.do_duty)
        self.client: BERTClient = client
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