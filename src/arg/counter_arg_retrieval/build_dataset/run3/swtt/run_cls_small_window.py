import sys
from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.ca_query import load_ca_query_from_tsv, CAQuery
from arg.counter_arg_retrieval.build_dataset.run3.swtt.run_classifier import run_batch_predictions
from bert_api.predictor import Predictor, FloatPredictor, PredictorWrap
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.window_enum_policy import WindowEnumPolicyMinPop
from cache import load_from_pickle


def main():
    model_path = sys.argv[1]
    tsv_path = sys.argv[2]
    ca_query_list: List[CAQuery] = load_ca_query_from_tsv(tsv_path)
    save_path = tsv_path + ".result"
    docs: List[Tuple[str, SegmentwiseTokenizedText]] = load_from_pickle("ca_run3_swtt")
    enum_policy = WindowEnumPolicyMinPop(30, 50)
    predictor: FloatPredictor = PredictorWrap(Predictor(model_path, 2), lambda x: x[1])
    run_batch_predictions(ca_query_list, docs, predictor, enum_policy, save_path)


if __name__ == "__main__":
    main()
