import argparse
import sys
from typing import Tuple

from explain.pairing.match_predictor import ProbeConfigI
from explain.pairing.probe.train_probe import train_probe
from explain.pairing.probe_train_common import NLIPairingTrainConfig
from explain.pairing.runner.run_train import HPCommon
from explain.pairing.runner_mmd.data_reader import read_tfrecord_as_triples, expand_input_files
from explain.setups import init_fn_generic
from tf_util.tf_logging import tf_logging, set_level_info, reset_root_log_handler
from trainer.np_modules import Batches

ex_arg_parser = argparse.ArgumentParser(description='File should be stored in ')
ex_arg_parser.add_argument("--start_model_path", help="Your input file.")
ex_arg_parser.add_argument("--start_type")
ex_arg_parser.add_argument("--save_dir")
ex_arg_parser.add_argument("--modeling_option")
ex_arg_parser.add_argument("--num_gpu", default=1)
ex_arg_parser.add_argument("--save_name")
ex_arg_parser.add_argument("--train_files")
ex_arg_parser.add_argument("--eval_files")


class ClsProbeConfig(ProbeConfigI):
    num_labels = 2
    use_embedding_out = True
    per_layer_component = 'linear'


class BertMMD:
    vocab_filename = "bert_voca.txt"
    vocab_size = 30522
    seq_length = 512


def get_mmd_data_from_tfrecord(train_files, eval_files, seq_length, batch_size) -> Tuple[Batches, Batches]:
    def read_tfrecord_to_batches(input_file_str):
        input_files = expand_input_files(input_file_str)
        dataset = read_tfrecord_as_triples(input_files, seq_length, batch_size, True)
        return iter(dataset)

    train_data = read_tfrecord_to_batches(train_files)
    eval_data = read_tfrecord_to_batches(eval_files)
    small_dev_batches = eval_data.take(20)
    return train_data, small_dev_batches


def main(start_model_path, start_type, save_dir,
         train_files, eval_files,
          modeling_option,
         num_gpu=1):
    set_level_info()
    num_gpu = int(num_gpu)
    tf_logging.info("run train MMD cls probe")
    tf_logging.info("train_from : {}".format(start_type))
    hp = HPCommon()
    hp.num_classes = 2
    task_setting = BertMMD()
    reset_root_log_handler()
    train_config = NLIPairingTrainConfig()
    train_config.num_gpu = num_gpu

    tf_logging.info("loading batches")
    data = get_mmd_data_from_tfrecord(train_files, eval_files, task_setting.seq_length, hp.batch_size)

    def init_fn(sess):
        return init_fn_generic(sess, start_type, start_model_path)
    cls_probe_config = ClsProbeConfig()
    cls_probe_config.per_layer_component = modeling_option
    train_probe(hp, train_config, cls_probe_config, save_dir, data, init_fn)


if __name__ == "__main__":
    args = ex_arg_parser.parse_args(sys.argv[1:])
    main(args.start_model_path,
         args.start_type,
         args.save_dir,
         args.train_files,
         args.eval_files,
         args.modeling_option,
         args.num_gpu)
