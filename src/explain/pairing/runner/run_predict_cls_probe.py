import sys

import tensorflow as tf

from cache import save_to_pickle
from explain.pairing.predict import predict_fn
from explain.pairing.predict_on_dev_common import prepare_predict_setup
from explain.pairing.probe_model import ClsProbeModel
from explain.pairing.runner.run_train_cls_probe import ClsProbeConfig
from explain.runner.nli_ex_param import ex_arg_parser
from explain.setups import init_fn_generic
from trainer.tf_train_module import init_session


def do_predict(bert_hp, train_config, dev_batches,
               lms_config, init_fn,
               ):
    num_gpu = train_config.num_gpu
    model = ClsProbeModel(bert_hp, lms_config, num_gpu, False)
    sess = init_session()
    sess.run(tf.global_variables_initializer())
    init_fn(sess)
    dummy_ex_score_tensor = tf.zeros([1, 1])
    # make explain train_op does not increase global step
    output_d = predict_fn(sess, dev_batches,
                          model.logits,
                          model.loss_tensor,
                          dummy_ex_score_tensor,
                          model.per_layer_logit_tensor, model.batch2feed_dict)
    return output_d


def main(start_model_path, modeling_option, save_name, num_gpu=1):
    dev_batches, hp, train_config = prepare_predict_setup(num_gpu)

    def init_fn(sess):
        return init_fn_generic(sess, "as_is", start_model_path)
    probe_config = ClsProbeConfig()
    probe_config.per_layer_component = modeling_option
    output_d = do_predict(hp, train_config, dev_batches,
                          probe_config, init_fn)
    save_to_pickle(output_d, save_name)


if __name__ == "__main__":
    args = ex_arg_parser.parse_args(sys.argv[1:])
    main(args.start_model_path,
         args.modeling_option,
         args.save_name,
         args.num_gpu)
