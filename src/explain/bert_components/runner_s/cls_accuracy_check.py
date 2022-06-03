from typing import List

import numpy as np
from scipy.special import softmax

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.mnli.mnli_reader import MNLIReader
from explain.bert_components.cls_probe_predictor import ClsProbePredictor
from misc_lib import SuccessCounter, TimeEstimator


def main():
    reader = MNLIReader()
    predictor = ClsProbePredictor()
    tokenizer = get_tokenizer()

    def enc(text) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    suc_count = SuccessCounter()
    ticker = TimeEstimator(50)
    for e in reader.load_split("dev"):
        output = predictor.predict(enc(e.premise), enc(e.hypothesis))
        probs = softmax(output.logits, axis=0)
        pred = np.argmax(probs)
        if pred == e.get_label_as_int():
            suc_count.suc()
        else:
            suc_count.fail()

        ticker.tick()
        if suc_count.n_total == 50:
            break

    print("accuracy : ", suc_count.get_suc_prob())

if __name__ == "__main__":
    main()