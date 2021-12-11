import os
from typing import List, Tuple

from bert_api.predictor import FloatPredictor
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.swtt_scorer_def import SWTTScorerInput, SWTTScorerOutput
from cpath import data_path
from misc_lib import TEL
from trainer.promise import PromiseKeeper, MyFuture, MyPromise


class DocumentScorerSWTT:
    # This class does three things
    # 1. Splitting SWTT into passages
    # 2. Convert SWTTScorerInput into score input
    # 3. Combine structured output

    def __init__(self,
                 predictor: FloatPredictor,
                 encoder_factory,
                 max_seq_length,
                 max_seg_per_document=None):
        self.max_seq_length = max_seq_length
        self.pk: PromiseKeeper = PromiseKeeper(self.do_duty)
        self.max_seg_per_document = max_seg_per_document
        self.predictor: FloatPredictor = predictor
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.encoder = encoder_factory(self.max_seq_length, voca_path)

    def do_duty(self, l: List[SWTTScorerInput]) -> List[SWTTScorerOutput]:
        def encode(t) -> Tuple[List, List, List]:
            return self.encoder.encode_token_pairs(*t)

        output_list: List[SWTTScorerOutput] = []
        for e in TEL(l):
            payload_list: List[Tuple[List[str], List[str]]] = e.payload_list
            triplet_list = list(map(encode, payload_list))
            scores: List[float] = self.predictor.predict(triplet_list)
            output: SWTTScorerOutput = SWTTScorerOutput(e.windows_st_ed_list,
                                                        scores,
                                                        e.doc)
            output_list.append(output)
        return output_list

    def score(self, q_tokens: List[str],
              doc: SegmentwiseTokenizedText,
              window_enum_fn) -> MyFuture[SWTTScorerOutput]:
        available_doc_len = self.max_seq_length - len(q_tokens) - 3
        window_list = window_enum_fn(doc, available_doc_len)

        payload_list: List[Tuple[List[str], List[str]]] = []
        for window_idx, window in enumerate(window_list):
            e: Tuple[List[str], List[str]] = (q_tokens, doc.get_window_sb_tokens(window))
            payload_list.append(e)

        promise_input: SWTTScorerInput = SWTTScorerInput(window_list, payload_list, doc)
        promise = MyPromise(promise_input, self.pk)
        return promise.future()

