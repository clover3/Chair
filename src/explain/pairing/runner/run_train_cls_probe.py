import sys

from data_generator.shared_setting import BertNLI
from explain.pairing.match_predictor import ProbeConfigI
from explain.pairing.probe_train_common import NLIPairingTrainConfig
from explain.pairing.runner.run_train import HPCommon
from explain.pairing.train_probe import train_probe
from explain.runner.nli_ex_param import ex_arg_parser
from explain.setups import init_fn_generic
from explain.train_nli import get_nli_data
from tf_util.tf_logging import tf_logging, set_level_info, reset_root_log_handler


class ClsProbeConfig(ProbeConfigI):
    num_labels = 3
    use_embedding_out = True
    per_layer_component = 'linear'


def main(start_model_path, start_type, save_dir, modeling_option,
         num_gpu=1):
    set_level_info()
    num_gpu = int(num_gpu)
    tf_logging.info("run train cls probe")
    tf_logging.info("train_from : {}".format(start_type))
    hp = HPCommon()
    nli_setting = BertNLI()
    reset_root_log_handler()
    train_config = NLIPairingTrainConfig()
    train_config.num_gpu = num_gpu

    tf_logging.info("loading batches")
    data = get_nli_data(hp, nli_setting)

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
         args.modeling_option,
         args.num_gpu)
