import os
from cpath import output_path
from tf_v2_support import disable_eager_execution

from bert_api.bert_mask_predictor import PredictorAttentionMask
from bert_api.bert_masking_common import BERTMaskIF


def get_mmd_z_bert_mask_predictor() -> BERTMaskIF:
    save_path = os.path.join(output_path, "model", "runs", "mmd_Z")
    disable_eager_execution()
    predictor = PredictorAttentionMask(2, 512)
    predictor.load_model(save_path)
    return predictor
