import sys

import numpy as np
import tensorflow as tf

from attribution.baselines import explain_by_seq_deletion, explain_by_random, explain_by_deletion
from attribution.eval import eval_fidelity
from attribution.lime import explain_by_lime
from data_generator.NLI import enlidef
from data_generator.shared_setting import BertNLI
from explain.nli_ex_predictor import NLIExPredictor
from explain.run_baselines import nli_ex_prediction_parser
from explain.train_nli import get_nli_data
from models.transformer import hyperparams
from models.transformer.nli_base import transformer_nli_pooled_embedding_in, transformer_nli_pooled
from trainer.model_saver import load_model
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module import init_session


def flatten_filter_batches(batches, target_class):
    output = []
    for batch in batches:
        x0, x1, x2, y = batch
        for i in range(len(x0)):
            if y[i] == target_class:
                output.append((x0[i], x1[i], x2[i]))
    return output


def eval_fidelity_gradient(hparam, nli_setting, flat_dev_batches,
                           explain_tag, method_name, model_path):

    from attribution.gradient import explain_by_gradient
    from attribution.deepexplain.tensorflow import DeepExplain

    sess = init_session()

    with DeepExplain(session=sess, graph=sess.graph) as de:

        task = transformer_nli_pooled_embedding_in(hparam, nli_setting.vocab_size, False)
        softmax_out = tf.nn.softmax(task.logits, axis=-1)
        sess.run(tf.global_variables_initializer())
        load_model(sess, model_path)
        emb_outputs = task.encoded_embedding_out, task.attention_mask_out
        emb_input = task.encoded_embedding_in, task.attention_mask_in

        def feed_end_input(batch):
            x0, x1, x2 = batch
            return {task.x_list[0]:x0,
                    task.x_list[1]:x1,
                    task.x_list[2]:x2,
                    }

        def forward_runs(insts):
            alt_batches = get_batches_ex(insts, hparam.batch_size, 3)
            alt_logits = []
            for batch in alt_batches:
                enc, att = sess.run(emb_outputs, feed_dict=feed_end_input(batch))
                logits, = sess.run([softmax_out, ],
                                   feed_dict={
                                       task.encoded_embedding_in: enc,
                                       task.attention_mask_in: att
                                   })

                alt_logits.append(logits)
            alt_logits = np.concatenate(alt_logits)
            return alt_logits

        contrib_score = explain_by_gradient(flat_dev_batches, method_name, explain_tag, sess, de,
                                       feed_end_input, emb_outputs, emb_input, softmax_out)
        print("contrib_score", len(contrib_score))
        print("flat_dev_batches", len(flat_dev_batches))

        acc_list = eval_fidelity(contrib_score, flat_dev_batches, forward_runs, explain_tag)

        return acc_list


class BaselineExPredictor:
    def __init__(self, hparam, nli_setting, model_path, method_name):
        self.task = transformer_nli_pooled(hparam, nli_setting.vocab_size, False)
        self.sout = tf.nn.softmax(self.task.logits, axis=-1)
        self.sess = init_session()
        self.sess.run(tf.global_variables_initializer())
        self.batch_size= hparam.batch_size
        load_model(self.sess, model_path)
        todo_list = [
            ('deletion_seq', explain_by_seq_deletion),
            ('random', explain_by_random),
            ('deletion', explain_by_deletion),
            ('LIME', explain_by_lime),
        ]
        method_dict = dict(todo_list)
        self.method = method_dict[method_name]

    def predict_ex_from_insts(self, explain_tag, insts):
        contrib_score = self.method(insts, explain_tag, self.forward_run)
        return contrib_score

    def forward_run(self, inputs):
        batches = get_batches_ex(inputs, self.batch_size, 3)
        logit_list = []
        task = self.task
        for batch in batches:
            x0, x1, x2 = batch
            soft_out, = self.sess.run([self.sout, ],
                                      feed_dict={
                                          task.x_list[0]: x0,
                                          task.x_list[1]: x1,
                                          task.x_list[2]: x2,
                                      })
            logit_list.append(soft_out)
        return np.concatenate(logit_list)


def eval_fidelity_with_ex_predictor(predictor, flat_dev_batches, explain_tag):
    contrib_score = predictor.predict_ex_from_insts(explain_tag, flat_dev_batches)
    print("contrib_score", len(contrib_score))
    print("flat_dev_batches", len(flat_dev_batches))
    acc_list = eval_fidelity(contrib_score, flat_dev_batches, predictor.forward_run, explain_tag)
    return acc_list


def run(explain_tag, method_name, model_path):
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    target_class = enlidef.get_target_class(explain_tag)
    data = get_nli_data(hp, nli_setting)
    train_batches, dev_batches = data
    flat_dev_batches = flatten_filter_batches(dev_batches, target_class)[:2000]

    if method_name in ['deletion_seq', "random", 'idf', 'deletion', 'LIME']:
        predictor = BaselineExPredictor(hp, nli_setting, model_path, method_name)
        acc_list = eval_fidelity_with_ex_predictor(predictor, flat_dev_batches, explain_tag)
    elif method_name.startswith("nli_ex"):
        modeling_option = "co"
        predictor = NLIExPredictor(hp, nli_setting, model_path, modeling_option)
        acc_list = eval_fidelity_with_ex_predictor(predictor, flat_dev_batches, explain_tag)
    elif method_name in ["elrp", "deeplift", "saliency","grad*input", "intgrad"]:
        acc_list = eval_fidelity_gradient(hp, nli_setting, flat_dev_batches,
                                    explain_tag, method_name, model_path)
    else:
        raise Exception("method_name={} is not in the known method list.".format(method_name))

    print(method_name)
    for num_delete in sorted(acc_list.keys()):
        print("{}\t{}".format(num_delete, acc_list[num_delete]))


if __name__ == "__main__":
    args = nli_ex_prediction_parser.parse_args(sys.argv[1:])
    run(args.tag, args.method_name, args.model_path)