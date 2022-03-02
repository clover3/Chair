import os
import random
from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.passage_scoring.hash_helper import get_int_list_tuples_hash
from arg.counter_arg_retrieval.build_dataset.passage_scoring.offline_scorer_bert_like import FileOfflineScorerBertLike, \
    to_feature
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.swtt_scorer_def import SWTTScorerInput, SWTTScorerOutput
from cpath import data_path, output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import DataIDManager, TEL, exist_or_mkdir
from trainer.promise import PromiseKeeper, MyFuture, MyPromise, list_future

Logit = List[float]

V = Tuple[SWTTScorerInput, List[MyFuture[Logit]]]


class OfflineSWTTScorerBertLike(FutureScorerI):
    def __init__(self,
                 signature,
                 encoder_factory,
                 max_seq_length,
                 logit_to_score,
                 ):
        self.max_seq_length = max_seq_length
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.signature = signature
        self.encoder = encoder_factory(self.max_seq_length, voca_path)
        self.logit_to_score = logit_to_score
        self.tokenizer = get_tokenizer()
        self.data_id_manager = DataIDManager()
        self.pk_inner = None
        self.pk_pre = PromiseKeeper(self._preprocess) # Input = SWTTScorerInput
        self.pk_post = PromiseKeeper(self._post_process) # Input = SWTTScorerInput

    def _preprocess(self, l: List[SWTTScorerInput]) -> List[V]:
        def encode(t) -> Tuple[List, List, List]:
            return self.encoder.encode_token_pairs(*t)

        def add_job_for_swtt(e: SWTTScorerInput) -> V:
            payload_list: List[Tuple[List[str], List[str]]] = e.payload_list
            triplet_list: List[Tuple[List, List, List]] = list(map(encode, payload_list))

            logits_future_list: List[MyFuture[Logit]] = []
            for triple_i_list in triplet_list:
                triple_i_list: Tuple[List[int], List[int], List[int]] = triple_i_list
                hash_val: bytes = get_int_list_tuples_hash(triple_i_list)
                data_id = self.data_id_manager.assign({'hash': hash_val})
                future = self.pk_inner.add_item(to_feature(triple_i_list), hash_val, data_id)
                logits_future_list.append(future)
            return e, logits_future_list

        return [add_job_for_swtt(e) for e in TEL(l)]

    def _post_process(self, items: List[V]) -> List[SWTTScorerOutput]:
        def do_for_item(item: V) -> SWTTScorerOutput:
            swtt_input: SWTTScorerInput = item[0]
            logits_future_list: List[MyFuture[Logit]] = item[1]
            logits_list: List[Logit] = list_future(logits_future_list)
            score_list: List[float] = [self.logit_to_score(logit) for logit in logits_list]
            return SWTTScorerOutput(swtt_input.windows_st_ed_list, score_list, swtt_input.doc)
        return [do_for_item(item) for item in items]

    def get_score_future(self, query_text: str,
                         doc: SegmentwiseTokenizedText,
                         passages: List[PassageRange]) -> MyFuture[SWTTScorerOutput]:
        q_tokens = self.tokenizer.tokenize(query_text)
        payload_list: List[Tuple[List[str], List[str]]] = []
        for window_idx, window in enumerate(passages):
            e: Tuple[List[str], List[str]] = (q_tokens, doc.get_window_sb_tokens(window))
            payload_list.append(e)
        promise_input: SWTTScorerInput = SWTTScorerInput(passages, payload_list, doc)
        promise = MyPromise(promise_input, self.pk_pre)
        return promise.future()

    def do_duty(self):
        save_path = get_offline_path(self.signature)
        self.pk_inner = FileOfflineScorerBertLike(save_path)
        self.pk_pre.do_duty(True, True) # This gives jobs to self.pk_inner
        self.pk_inner.do_duty()
        self.pk_post.do_duty(True, True)


def get_offline_path(readable_id):
    offline_dir = os.path.join(output_path, "offline")
    exist_or_mkdir(offline_dir)
    save_path = os.path.join(offline_dir, "{}_{}".format(readable_id, random.randint(0, 9999)))
    return save_path