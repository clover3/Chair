import collections
import os
import random

import numpy as np

from data_generator.job_runner import JobRunner
from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature, create_float_feature

working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"


class Worker:
    def __init__(self, working_path):
        self.working_path = working_path
        self.max_seq_length = 512
        self.inner_batch_size = 32
        self.max_predictions_per_seq = 20

    def work(self, job_id):
        return self.work_inner(os.path.join(working_path, "ssdr_dummy3", str(job_id)),
                               os.path.join(self.working_path, str(job_id)))

    def work_inner(self, input_file_path, output_path):

        itr = load_record(input_file_path)
        writer = RecordWriterWrap(output_path)

        def reform_a_input(raw_input):
            return np.reshape(raw_input, [self.inner_batch_size, self.max_seq_length])

        def reform_mask_input(raw_input):
            return np.reshape(raw_input, [self.inner_batch_size, self.max_predictions_per_seq])

        def get_as_list(feature, name):
            ids = list(feature[name].int64_list.value)
            ids_list = reform_a_input(ids)
            return ids_list

        all_features = []
        for feature in itr:
            listed_inputs = {}
            for key in ["input_ids", "input_mask", "segment_ids"]:
                listed_inputs[key] = get_as_list(feature, key)

            for key in ["masked_lm_positions","masked_lm_ids"]:
                ids = list(feature[key].int64_list.value)
                listed_inputs[key] = reform_mask_input(ids)

            listed_inputs["masked_lm_weights"] = reform_mask_input(feature["masked_lm_weights"].float_list.value)

            for i in range(self.inner_batch_size):
                new_features = collections.OrderedDict()
                for key, value in listed_inputs.items():
                    if key is "masked_lm_weights":
                        new_features[key] = create_float_feature(value[i])
                    else:
                        new_features[key] = create_int_feature(value[i])
                all_features.append(new_features)

        random.shuffle(all_features)
        for f in all_features:
            writer.write_feature(f)
        writer.close()


if __name__ == "__main__":
    runner = JobRunner(working_path, 500, "ssdr_dummy3_flatten", Worker)
    runner.start()


