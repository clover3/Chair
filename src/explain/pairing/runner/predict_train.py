import sys

import tensorflow as tf

from cache import save_to_pickle
from cpath import at_output_dir
from data_generator.shared_setting import BertNLI
from explain.pairing.lms_model import LMSModel
from explain.pairing.match_predictor import LMSConfig
from explain.pairing.predict import predict_fn
from explain.pairing.probe_train_common import NLIPairingTrainConfig
from explain.pairing.runner.run_train import HPCommon
from explain.runner.nli_ex_param import ex_arg_parser
from explain.setups import init_fn_generic
from explain.train_nli import get_nli_data
from misc_lib import tprint
from tf_util.tf_logging import set_level_debug, reset_root_log_handler
from trainer.tf_train_module import init_session


def do_predict(bert_hp, train_config, data,
               lms_config, modeling_option, init_fn,
               ):
    num_gpu = train_config.num_gpu
    train_batches, dev_batches = data

    lms_model = LMSModel(modeling_option, bert_hp, lms_config, num_gpu)
    sess = init_session()
    sess.run(tf.global_variables_initializer())
    init_fn(sess)

    step_size = 100
    for i in range(100):
        st = i * step_size
        ed = st + step_size
        # make explain train_op does not increase global step
        tprint(st, ed)
        output_d = predict_fn(sess, train_batches[st:ed], lms_model.logits, lms_model.loss_tensor, lms_model.ex_score_tensor,
                              lms_model.per_layer_logit_tensor, lms_model.batch2feed_dict)

        save_path = at_output_dir("lms_scores", str(i))
        save_to_pickle(output_d, save_path)


def main(start_model_path, modeling_option, num_gpu=1):
    num_gpu = int(num_gpu)
    hp = HPCommon()
    nli_setting = BertNLI()
    set_level_debug()
    reset_root_log_handler()
    train_config = NLIPairingTrainConfig()
    train_config.num_gpu = num_gpu

    def init_fn(sess):
        return init_fn_generic(sess, "as_is", start_model_path)
    data = get_nli_data(hp, nli_setting)
    do_predict(hp, train_config, data, LMSConfig(), modeling_option, init_fn)


if __name__ == "__main__":
    args = ex_arg_parser.parse_args(sys.argv[1:])
    main(args.start_model_path,
               args.modeling_option,
               args.num_gpu)
