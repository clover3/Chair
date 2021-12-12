import os
from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.run3.run_interface.passage_scorer import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from bert_api.predictor import FloatPredictor
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.swtt_scorer_def import SWTTScorerInput, SWTTScorerOutput
from cpath import data_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import TEL
from trainer.promise import PromiseKeeper, MyFuture, MyPromise


class FutureScorerBertLike(FutureScorerI):
    def __init__(self,
                 predictor: FloatPredictor,
                 encoder_factory,
                 max_seq_length
                 ):
        self.predictor: FloatPredictor = predictor
        self.max_seq_length = max_seq_length
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.encoder = encoder_factory(self.max_seq_length, voca_path)
        self.pk = PromiseKeeper(self._score_inner)
        self.tokenizer = get_tokenizer()

    def _score_inner(self, l: List[SWTTScorerInput]) -> List[SWTTScorerOutput]:
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

    def get_score_future(self, query_text: str,
                         doc: SegmentwiseTokenizedText,
                         passages: List[PassageRange]) -> MyFuture[SWTTScorerOutput]:
        q_tokens = self.tokenizer.tokenize(query_text)
        payload_list: List[Tuple[List[str], List[str]]] = []
        for window_idx, window in enumerate(passages):
            e: Tuple[List[str], List[str]] = (q_tokens, doc.get_window_sb_tokens(window))
            payload_list.append(e)
        promise_input: SWTTScorerInput = SWTTScorerInput(passages, payload_list, doc)
        promise = MyPromise(promise_input, self.pk)
        return promise.future()

    def do_duty(self):
        self.pk.do_duty()