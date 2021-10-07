import argparse
import os
import sys

from cache import save_to_pickle
from cpath import data_path
from data_generator.light_dataloader import LightDataLoader
from data_generator.shared_setting import BertNLI
from explain.pairing.probe_train_common import HPCommon, NLIPairingTrainConfig
from explain.pairing.runner.run_predict_cls_probe import do_predict
from explain.pairing.runner.run_train_cls_probe import ClsProbeConfig
from explain.setups import init_fn_generic
from trainer.np_modules import get_batches_ex


def get_batches(file_path, nli_setting: BertNLI, batch_size):
    voca_path = os.path.join(data_path, nli_setting.vocab_filename)
    data_loader = LightDataLoader(nli_setting.seq_length, voca_path)
    data = list(data_loader.example_generator(file_path))
    return get_batches_ex(data, batch_size, 4)


def main(start_model_path, modeling_option, input_path, save_name, num_gpu=1):
    num_gpu = int(num_gpu)
    hp = HPCommon()
    nli_setting = BertNLI()
    train_config = NLIPairingTrainConfig()
    train_config.num_gpu = num_gpu

    def init_fn(sess):
        return init_fn_generic(sess, "as_is", start_model_path)
    probe_config = ClsProbeConfig()
    probe_config.per_layer_component = modeling_option
    batches = get_batches(input_path, nli_setting, HPCommon.batch_size)
    output_d = do_predict(hp, train_config, batches,
                          probe_config, init_fn)
    save_to_pickle(output_d, save_name)


ex_arg_parser = argparse.ArgumentParser(description='File should be stored in ')
ex_arg_parser.add_argument("--start_model_path", help="Your input file.")
ex_arg_parser.add_argument("--modeling_option")
ex_arg_parser.add_argument("--num_gpu", default=1)
ex_arg_parser.add_argument("--input_path")
ex_arg_parser.add_argument("--save_name")


if __name__ == "__main__":
    args = ex_arg_parser.parse_args(sys.argv[1:])
    main(args.start_model_path,
         args.modeling_option,
         args.input_path,
         args.save_name,
         args.num_gpu)
