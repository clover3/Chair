import argparse
import sys

import tensorflow as tf

from cache import save_to_pickle
from data_generator.shared_setting import BertNLI
from explain.pairing.lms_model import LMSModel
from explain.pairing.match_predictor import LMSConfig
from explain.pairing.pair_deletion_common import get_payload
from explain.pairing.predict import predict_fn
from explain.pairing.probe_train_common import NLIPairingTrainConfig
from explain.pairing.runner.run_train import HPCommon
from explain.setups import init_fn_generic
from tf_util.tf_logging import set_level_debug, reset_root_log_handler
from trainer.tf_train_module import init_session


def do_predict(bert_hp, train_config, batches,
               lms_config, modeling_option, init_fn,
               ):
    num_gpu = train_config.num_gpu

    lms_model = LMSModel(modeling_option, bert_hp, lms_config, num_gpu)
    sess = init_session()
    sess.run(tf.global_variables_initializer())
    init_fn(sess)

    # make explain train_op does not increase global step
    output_d = predict_fn(sess, batches, lms_model.logits, lms_model.loss_tensor, lms_model.ex_score_tensor,
                          lms_model.per_layer_logit_tensor, lms_model.batch2feed_dict)
    return output_d


def main(start_model_path, modeling_option, input_path, save_name, num_gpu=1):
    num_gpu = int(num_gpu)
    hp = HPCommon()
    nli_setting = BertNLI()
    set_level_debug()
    reset_root_log_handler()
    train_config = NLIPairingTrainConfig()
    train_config.num_gpu = num_gpu

    def init_fn(sess):
        return init_fn_generic(sess, "as_is", start_model_path)
    batches, info = get_payload(input_path, nli_setting, hp.batch_size)
    output_d = do_predict(hp, train_config, batches, LMSConfig(), modeling_option, init_fn)
    result = info, output_d
    save_to_pickle(result, save_name)


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
