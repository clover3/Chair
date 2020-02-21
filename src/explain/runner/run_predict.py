import sys

import numpy as np
import tensorflow as tf

from attribution.eval import predict_translate
from cache import save_to_pickle
from data_generator.NLI.nli import get_modified_data_loader2, tags
from data_generator.shared_setting import BertNLI
from explain.explain_trainer import ExplainPredictor
from explain.runner.predict_params import parser
from models.transformer import hyperparams
from models.transformer.nli_base import transformer_nli_pooled
from trainer.model_saver import load_model
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module import init_session


class NLIExPredictor:
    def __init__(self, hparam, nli_setting, model_path, modeling_option):
        self.define_graph(hparam, nli_setting, modeling_option)
        self.sess = init_session()
        self.sess.run(tf.global_variables_initializer())
        self.batch_size = hparam.batch_size
        load_model(self.sess, model_path)

    def define_graph(self, hparam, nli_setting, modeling_option):
        self.task = transformer_nli_pooled(hparam, nli_setting.vocab_size, False)
        self.sout = tf.nn.softmax(self.task.logits, axis=-1)
        self.explain_predictor = ExplainPredictor(len(tags), self.task.model.get_sequence_output(), modeling_option)


    def predict_ex(self, explain_tag, batches):
        tag_idx = tags.index(explain_tag)
        ex_logits_list = []
        for batch in batches:
            x0, x1, x2 = batch
            ex_logits, = self.sess.run([self.explain_predictor.get_score()[tag_idx]],
                                  feed_dict={
                                      self.task.x_list[0]: x0,
                                      self.task.x_list[1]: x1,
                                      self.task.x_list[2]: x2,
                                  })
            ex_logits_list.append(ex_logits)
        ex_logits = np.concatenate(ex_logits_list)
        return ex_logits

    def predict_ex_from_insts(self, explain_tag, insts):
        batches = get_batches_ex(insts, self.batch_size, 3)
        return self.predict_ex(explain_tag, batches)

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



def predict_nli_ex(hparam, nli_setting, data_loader,
                   explain_tag, data_id, model_path, run_name, modeling_option):
    print("predict_nli_ex")
    print("Modeling option: ", modeling_option)
    enc_payload, plain_payload = data_loader.load_plain_text(data_id)
    batches = get_batches_ex(enc_payload, hparam.batch_size, 3)

    predictor = NLIExPredictor(hparam, nli_setting, model_path, modeling_option)
    ex_logits = predictor.predict_ex(explain_tag, batches)
    pred_list = predict_translate(ex_logits, data_loader, enc_payload, plain_payload)
    save_to_pickle(pred_list, "pred_{}_{}".format(run_name, data_id))


def run(args):
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    data_loader = get_modified_data_loader2(hp, nli_setting)

    predict_nli_ex(hp, nli_setting, data_loader,
                         args.tag,
                         args.data_id,
                         args.model_path,
                         args.run_name,
                         args.modeling_option,
                         )

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    run(args)