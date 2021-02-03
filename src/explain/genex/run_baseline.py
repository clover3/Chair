import argparse
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from attribution.baselines import explain_by_seq_deletion, explain_by_random, IdfScorer, explain_by_deletion
from attribution.lime import explain_by_lime
from cache import save_to_pickle
from data_generator.shared_setting import BertNLI
from explain.genex.load import load_as_simple_format
from explain.nli_gradient_baselines import nli_attribution_predict
from explain.train_nli import get_nli_data
from misc_lib import tprint
from models.transformer import hyperparams
from models.transformer.transfomer_logit import transformer_logit
from trainer.model_saver import load_model
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module import init_session

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument("--data_name", help="data_name")
arg_parser.add_argument("--model_path", help="Your model path.")
arg_parser.add_argument("--method_name", )


def baseline_predict(hparam, nli_setting, data, method_name, model_path) -> List[np.array]:
    tprint("building model")
    voca_size = 30522
    task = transformer_logit(hparam, 2, voca_size, False)
    enc_payload: List[Tuple[np.array, np.array, np.array]] = data

    sout = tf.nn.softmax(task.logits, axis=-1)
    sess = init_session()
    sess.run(tf.global_variables_initializer())

    tprint("loading model")
    load_model(sess, model_path)

    def forward_run(inputs):
        batches = get_batches_ex(inputs, hparam.batch_size, 3)
        logit_list = []
        for batch in batches:
            x0, x1, x2 = batch
            soft_out, = sess.run([sout, ],
                                  feed_dict={
                                      task.x_list[0]: x0,
                                      task.x_list[1]: x1,
                                      task.x_list[2]: x2,
                                  })
            logit_list.append(soft_out)
        return np.concatenate(logit_list)

    # train_batches, dev_batches = self.load_nli_data(data_loader)
    def idf_explain(enc_payload, explain_tag, forward_run):
        train_batches, dev_batches = get_nli_data(hparam, nli_setting)
        idf_scorer = IdfScorer(train_batches)
        return idf_scorer.explain(enc_payload, explain_tag, forward_run)

    todo_list = [
        ('deletion_seq', explain_by_seq_deletion),
        ('random', explain_by_random),
        ('idf', idf_explain),
        ('deletion', explain_by_deletion),
        ('LIME', explain_by_lime),
    ]
    method_dict = dict(todo_list)
    method = method_dict[method_name]
    explain_tag = "mismatch"
    explains: List[np.array] = method(enc_payload, explain_tag, forward_run)
    # pred_list = predict_translate(explains, data_loader, enc_payload, plain_payload)
    return explains


def run(args):
    hp = hyperparams.HPGenEx()
    nli_setting = BertNLI()

    if args.method_name in ['deletion_seq', "random", 'idf', 'deletion', 'LIME']:
        predictor = baseline_predict
    elif args.method_name in ["elrp", "deeplift", "saliency", "grad*input", "intgrad"]:
        predictor = nli_attribution_predict
    else:
        raise Exception("method_name={} is not in the known method list.".format(args.method_name))

    save_name = "{}_{}".format(args.data_name, args.method_name)
    data = load_as_simple_format(args.data_name)
    explains: List[np.array] = predictor(hp, nli_setting, data, args.method_name, args.model_path)

    save_to_pickle(explains, save_name)


if __name__ == "__main__":
    args = arg_parser.parse_args(sys.argv[1:])
    run(args)
