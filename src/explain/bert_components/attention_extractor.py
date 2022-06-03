from typing import List

import numpy as np

from cpath import pjoin, data_path
from data_generator.tokenizer_wo_tf import get_tokenizer, EncoderUnitPlain
from explain.bert_components.cmd_nli import ModelConfig
from explain.bert_components.inspections.attention_summary import execute_with_attention_logging
from trainer.np_modules import get_batches_ex


class AttentionExtractor:
    def __init__(self):
        model_config = ModelConfig()
        from explain.bert_components.cls_probe_w_visualize import load_probe
        model, bert_cls_probe = load_probe(model_config)
        self.bert_cls_probe = bert_cls_probe
        voca_path = pjoin(data_path, "bert_voca.txt")
        self.d_encoder = EncoderUnitPlain(model_config.max_seq_length, voca_path)

    def predict(self, p_tokens_id, h_tokens_id):
        d = self.d_encoder.encode_inner(p_tokens_id, h_tokens_id)
        layers = {}

        def hooking_fn(layer_no, attention_probs):
            layers[layer_no] = attention_probs[0]

        single_x = d["input_ids"], d["input_mask"], d["segment_ids"]
        X = get_batches_ex([single_x], 1, 3)[0]
        y = 0
        execute_with_attention_logging(self.bert_cls_probe, X, y, hooking_fn)

        attention_probs = np.stack([layers[i] for i in range(12)], axis=0)
        return attention_probs


def main():
    p = "this is premise"
    h = "this is hypothesis"

    tokenizer = get_tokenizer()

    def enc(text) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    ae = AttentionExtractor()
    ae.predict(enc(p), enc(h))


if __name__ == "__main__":
    main()