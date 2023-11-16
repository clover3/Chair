import pickle
import numpy as np
import os
import sys
from typing import Iterable, Callable

from list_lib import lflatten
from port_info import LOCAL_DECISION_PORT

from transformers import AutoTokenizer

from data_generator2.segmented_enc.es_common.es_two_seg_common import BothSegPartitionedPair
from data_generator2.segmented_enc.es_common.partitioned_encoder import get_both_seg_partitioned_to_input_ids2
from data_generator2.segmented_enc.es_mmp.pep_attn_common import iter_attention_data_pair
from misc_lib import path_join
from trainer.promise import PromiseKeeper, unpack_future, MyFuture
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.custom_loop.per_task.local_decision_server import get_local_decision_predictor
from trainer_v2.evidence_selector.environment import PEPClient, PEPClientFromPredictor
from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.compute_deletions_common import RandomSeqSelector, \
    delete_compute_pep
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def get_pep_client():
    server = "localhost"
    if "PEP_SERVER" in os.environ:
        server = os.environ["PEP_SERVER"]
    c_log.info("PEP_SERVER: {}".format(server))
    pep_client = PEPClient(server, LOCAL_DECISION_PORT)
    return pep_client


# mmp/pe_del.sh

def main():
    c_log.info(__file__)
    save_dir = sys.argv[1]
    model_path = sys.argv[2]
    part_no = int(sys.argv[3])

    max_seq_length = 256
    model_config = ModelConfig512_1()
    strategy = get_strategy()
    predict = get_local_decision_predictor(model_path, model_config, strategy)
    client = PEPClientFromPredictor(predict)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    EncoderType = Callable[[BothSegPartitionedPair], Iterable[tuple[list, list]]]
    encoder_iter: EncoderType = get_both_seg_partitioned_to_input_ids2(tokenizer, max_seq_length)

    def pep_client_request(candidate_itr: Iterable[BothSegPartitionedPair]) -> np.array:
        # each BothSegPartitionedPair generate two items, for part 0/1
        input_seg_ids: list[tuple[list, list]] = lflatten(map(encoder_iter, candidate_itr))
        raw_ret: list[list[float]] = client.request(input_seg_ids)
        scores_np = np.array(raw_ret)
        scores_np = np.reshape(scores_np, [-1, 2])
        return scores_np

    g = 0.5
    deleter = RandomSeqSelector(g)
    save_batch_size = 500
    max_target_len = 32

    pk = PromiseKeeper(pep_client_request)
    itr = iter_attention_data_pair(part_no)
    save_items = []
    batch_idx = 0
    for pair_data, _attn in itr:
        save_item_per_qd_future: tuple[list[BothSegPartitionedPair], list[MyFuture[np.array]]] = \
            delete_compute_pep(deleter, pk, tokenizer, max_target_len, pair_data)
        save_items.append(save_item_per_qd_future)
        if len(save_items) >= save_batch_size:
            save_path = path_join(save_dir, f"{part_no}_{batch_idx}")
            if os.path.exists(save_path):
                pass
            else:
                pk.do_duty(log_size=True)
                save_items: list[tuple[list[BothSegPartitionedPair], list[np.array]]]\
                    = unpack_future(save_items)
                pickle.dump(save_items, open(save_path, "wb"))
            batch_idx += 1
            save_items = []



if __name__ == "__main__":
    main()
