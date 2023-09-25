from arg.counter_arg_retrieval.build_dataset.passage_scoring.hash_helper import get_int_list_hash
from misc_lib import tensor_to_list
from trainer_v2.reinforce.defs import RLStateI


class RLStateTensor(RLStateI):
    def __init__(self, input_ids, segment_ids, input_ids_np, segment_ids_np):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.input_ids_np = input_ids_np
        self.segment_ids_np = segment_ids_np

    def input_ids_hash(self):
        items = self.input_ids_np.tolist()
        non_zero = [t for t in items if t != 0]
        return get_int_list_hash(non_zero)

        # return ", ".join(map(str, non_zero))
