import os
from cpath import output_path
from tf_v2_support import disable_eager_execution

from bert_api.bert_mask_predictor import PredictorAttentionMask
from bert_api.bert_masking_common import BERTMaskIF


def get_nli_bert_mask_predictor(save_path) -> BERTMaskIF:
    disable_eager_execution()
    predictor = PredictorAttentionMask(3, 300)
    predictor.load_model(save_path)
    return predictor
