import json
from abc import ABC, abstractmethod
from typing import List, Tuple
from typing import NamedTuple

import numpy as np
import scipy.special

from cpath import pjoin, data_path, at_output_dir
from data_generator.bert_input_splitter import get_sep_loc
from data_generator.tokenizer_wo_tf import EncoderUnitPlain
from datastore.sql_based_cache_client import SQLBasedCacheClientS, SQLBasedCacheClientStr
from explain.bert_components.cmd_nli import ModelConfig
from explain.bert_components.load_probe import load_probe
from tlm.qtype.partial_relevance.cache_db import bulk_save_s, build_db_s
from trainer.np_modules import get_batches_ex


class ProbeOutput(NamedTuple):
    logits: np.array
    probes: List[np.array]
    sep_idx1: int
    sep_idx2: int

    def get_layer_probe(self, layer_no):
        return scipy.special.softmax(self.probes[layer_no], axis=1)

    # return np.array([seq_length, num_classes])
    def get_avg_probe(self):
        probes_logits = np.array(self.probes[1:])
        assert len(probes_logits.shape) == 3
        probes_probs = scipy.special.softmax(probes_logits, axis=2)
        return np.mean(probes_probs, axis=0)

    def slice_seg1(self, arr):
        return arr[1:self.sep_idx1]

    def slice_seg2(self, arr):
        return arr[self.sep_idx1 + 1: self.sep_idx2]

    def get_seg2_prob(self, layer_no):
        return self.slice_seg2(self.get_layer_probe(layer_no))

    def to_json(self):
        return {
            'logits': self.logits.tolist(),
            'probes': [p.tolist() for p in self.probes],
            'sep_idx1': self.sep_idx1,
            'sep_idx2': self.sep_idx2,
        }

    @classmethod
    def from_json(cls, j):
        probes: List[np.array] = [np.array(l) for l in j['probes']]
        return ProbeOutput(np.array(j['logits']), probes, j['sep_idx1'], j['sep_idx2'])


class ClsProbePredictorIF(ABC):
    @abstractmethod
    def predict(self, p_tokens_id, h_tokens_id) -> ProbeOutput:
        pass


class ClsProbePredictor(ClsProbePredictorIF):
    def __init__(self):
        model_config = ModelConfig()
        voca_path = pjoin(data_path, "bert_voca.txt")
        self.d_encoder = EncoderUnitPlain(model_config.max_seq_length, voca_path)
        model, bert_cls_probe = load_probe(model_config)
        self.model = model
        self.bert_cls_probe = bert_cls_probe

    def predict(self, p_tokens_id, h_tokens_id):
        d = self.d_encoder.encode_inner(p_tokens_id, h_tokens_id)
        single_x = d["input_ids"], d["input_mask"], d["segment_ids"]
        X = get_batches_ex([single_x], 1, 3)[0]
        logits, probes = self.bert_cls_probe(X)
        probes = [np.array(p[0]) for p in probes]

        sep_idx1, sep_idx2 = get_sep_loc(d["input_ids"])
        # probes: List[ np.array(batch_size, seq_length, num_classes) ]
        return ProbeOutput(np.array(logits[0]), probes, sep_idx1, sep_idx2)


def get_cls_probe_cache_sqlite_path():
    return at_output_dir("sqlite_cache", "cls_probe_cache.sqlite")


class CacheClsProbePredictor(ClsProbePredictorIF):
    def __init__(self):
        predictor = ClsProbePredictor()

        def forward_fn(items: List[Tuple[List[int], List[int]]]) -> List[str]:
            def do_for_item(ab) -> str:
                a, b = ab
                output = predictor.predict(a, b)
                return json.dumps(output.to_json())

            return [do_for_item(i) for i in items]

        cache_client = SQLBasedCacheClientStr(forward_fn,
                                              str,
                                              0.035,
                                              get_cls_probe_cache_sqlite_path())
        self.cache_client = cache_client

    def predict(self, p_tokens_id, h_tokens_id) -> ProbeOutput:
        outs = self.cache_client.predict([(p_tokens_id, h_tokens_id)])
        output: str = outs[0]
        return ProbeOutput.from_json(json.loads(output))


def convert():
    def forward_fn(items: List[Tuple[List[int], List[int]]]) -> List[str]:
        return ["" for _ in items]

    cache_client = SQLBasedCacheClientS(forward_fn,
                                        str,
                                        0.035,
                                        get_cls_probe_cache_sqlite_path())

    dictionary = cache_client.volatile_cache_client.dictionary

    new_d = {}
    for k, v in dictionary.items():
        json_str = json.dumps(v)
        new_d[k] = json_str

    save_path = get_cls_probe_cache_sqlite_path() + ".new"
    build_db_s(save_path)
    bulk_save_s(save_path, new_d)


def main():
    def forward_fn(items: List[Tuple[List[int], List[int]]]) -> List[str]:
        return ["" for _ in items]

    cache_client = SQLBasedCacheClientStr(forward_fn,
                                          str,
                                          0.035,
                                          get_cls_probe_cache_sqlite_path() + ".new")


if __name__ == "__main__":
    main()
