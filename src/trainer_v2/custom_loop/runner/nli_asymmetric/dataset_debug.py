import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.neural_network_def.asymmetric import ModelConfig2SegProject
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser
import tensorflow as tf

def main(args):
    c_log.info("Start {}".format(__file__))
    run_config: RunConfig2 = get_run_config2_nli(args)

    model_config = ModelConfig2SegProject()
    seq_length_list = [model_config.max_seq_length1, model_config.max_seq_length2]

    print("seq_length_list", seq_length_list)
    dataset = get_two_seg_data(run_config.dataset_config.eval_files_path, run_config, model_config, True)
    for e in dataset:
        input1, input2 = e
        print(input1)
        print(input2)
        print(len(e))
        break


def validate_data(fn):
    keys = None
    length_d ={}
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        if keys is None:
            keys = list(feature.keys())
        for key in keys:
            if key == "masked_lm_weights":
                v = feature[key].float_list.value
            else:
                v = feature[key].int64_list.value

            print(key, len(v))
            if key not in length_d:
                length_d[key] = len(v)
            if length_d[key] == len(v):
                pass
            else:

                print("Error at {} ({}): {} != {}".format(fn, key, length_d[key], len(v)))
    print(keys)




if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
