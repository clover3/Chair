from bert_api.bert_masking_common import BERTMaskIF
from bert_api.task_clients.bert_masking_client import get_localhost_bert_mask_client
from bert_api.task_clients.mmd_z_interface.mmd_z_mask_predictor import get_mmd_z_bert_mask_predictor


def get_mmd_z_mask_client(option) -> BERTMaskIF:
    if option == "localhost":
        predictor: BERTMaskIF = get_localhost_bert_mask_client()
    elif option == "direct":
        print("use direct predictor")
        predictor: BERTMaskIF = get_mmd_z_bert_mask_predictor()
    else:
        raise ValueError()
    return predictor