import os
import random
from cache import load_pickle_from, save_list_to_jsonl_w_fn
from data_generator.NLI.nli_info import nli_tokenized_path
from misc_lib import TimeEstimator, ceil_divide
from port_info import KERAS_NLI_PORT
from trainer_v2.custom_loop.per_task.cip.cip_common import get_nli_baseline_pred_save_path, Prediction
from trainer_v2.keras_server.bert_like_client import BERTClient
from typing import List, Tuple


def main():
    random.seed(0)
    split = "train"

    def iter_result():
        data: List[Tuple[List[int], List[int], int]] = load_pickle_from(nli_tokenized_path(split))
        base_seq_length = 300
        nli_server_addr = os.environ['NLI_ADDR']
        nli_client: BERTClient = BERTClient(nli_server_addr, KERAS_NLI_PORT, base_seq_length)
        query_batch_size = 64
        n_item = 400 * 1000
        ticker = TimeEstimator(ceil_divide(n_item, query_batch_size))
        cursor = 0
        all_save_entries: List = []
        while cursor < len(data):
            data_slice = data[cursor: cursor+query_batch_size]

            payload = [(h, p) for h,p,l in data_slice]
            pred_list = nli_client.request_multiple_from_ids_pairs(payload)
            for phl, pred in zip(data_slice, pred_list):
                p, h, l = phl
                out_e = Prediction(p, h, l, pred)
                yield out_e
            cursor += query_batch_size
            ticker.tick()

    save_list_to_jsonl_w_fn(iter_result(),
                            get_nli_baseline_pred_save_path(split),
                            Prediction.to_json)


if __name__ == "__main__":
    main()