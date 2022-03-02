import collections
from typing import List, Dict
from typing import OrderedDict

from estimator_helper.output_reader import join_prediction_with_info
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature, create_byte_feature
#    Output:
#       TFRecord w data id
# Stage 2 - TPU Server
#    Output: Pickle : {'data_id': score}
# Stage 3 - Receive
from trainer.promise import MyFuture


# Stage 1 - Request
#    Input: Triple list
#       add Future:
#    do_duty()
#       halt


def to_feature(triple):
    feature = collections.OrderedDict()
    input_ids, input_mask, segment_ids = triple
    feature['input_ids'] = create_int_feature(input_ids)
    feature['input_mask'] = create_int_feature(input_mask)
    feature['segment_ids'] = create_int_feature(segment_ids)
    return feature


def wait_by_console_input():
    s = "no"
    while s != "done":
        print("Enter done if done")
        s = input()


def is_valid(d):
    return d['hash'] == d['input_hash']


class FileOfflineScorerBertLike:
    def __init__(self, save_path):
        self.save_path = save_path
        self.writer = RecordWriterWrap(save_path)
        self.info_d: Dict[str, Dict] = {}
        self.future_d: Dict[int, MyFuture] = {}

    def add_item(self, feature: OrderedDict, hash_val: bytes, data_id: int) -> MyFuture:
        self.info_d[str(data_id)] = {'input_hash': hash_val}
        feature['data_id'] = create_int_feature([data_id])
        feature['hash'] = create_byte_feature(hash_val)
        self.writer.write_feature(feature)
        future = MyFuture()
        self.future_d[data_id] = future
        return future

    def do_duty(self):
        self.writer.close()
        print("Now upload the file to gs and get prediction, save to ")
        print("Payload path: ", self.save_path)
        output_save_path = self.save_path + ".future"
        print("Put output at: ", output_save_path)
        wait_by_console_input()
        fetch_field_list = ["data_id", "logits", "hash"]
        data: List[Dict] = join_prediction_with_info(output_save_path, self.info_d, fetch_field_list)
        self.verify_data(data)

        for item in data:
            future = self.future_d[int(item['data_id'])]
            future.set_value(item['logits'])

    def verify_data(self, data):
        # Compare data with
        valid_list: List[Dict] = [d for d in data if is_valid(d)]
        n_invalid = len(data) - len(valid_list)
        if n_invalid:
            raise Exception()
        if len(data) != len(self.future_d):
            raise Exception()
