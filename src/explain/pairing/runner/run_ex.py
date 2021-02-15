import sys

from data_generator.shared_setting import BertNLI
from explain.pairing.match_predictor import LMSConfig2
from explain.pairing.train_pairing import NLIPairingTrainConfig, train_LMS, HPCommon
from explain.runner.nli_ex_param import ex_arg_parser
from explain.setups import init_fn_generic
from explain.train_nli import get_nli_data
from tf_util.tf_logging import tf_logging, set_level_debug, reset_root_log_handler


def main(start_model_path, start_type, save_dir,
         modeling_option, num_gpu=1):
    num_gpu = int(num_gpu)
    tf_logging.info("train_from : nli_ex")
    hp = HPCommon()
    nli_setting = BertNLI()
    set_level_debug()
    reset_root_log_handler()
    train_config = NLIPairingTrainConfig()
    train_config.num_gpu = num_gpu

    tf_logging.info("loading batches")
    data = get_nli_data(hp, nli_setting)

    def init_fn(sess):
        return init_fn_generic(sess, start_type, start_model_path)

    train_LMS(hp, train_config, LMSConfig2(), save_dir, data, modeling_option, init_fn)


if __name__ == "__main__":
    args = ex_arg_parser.parse_args(sys.argv[1:])
    main(args.start_model_path,
         args.start_type,
         args.save_dir,
         args.modeling_option,
         args.num_gpu)
