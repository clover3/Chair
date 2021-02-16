from typing import List, Tuple

import numpy as np
import tensorflow as tf

from attribution.baselines import IdfScorer, explain_by_seq_deletion, explain_by_random, explain_by_deletion
from attribution.lime import explain_by_lime
from explain.genex.deletion_ex import explain_by_term_deletion, explain_by_replace, \
    explain_by_term_replace
from explain.train_nli import get_nli_data
from misc_lib import tprint, TimeEstimator
from models.transformer.transfomer_logit import transformer_logit
from trainer.model_saver import load_model
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module import init_session


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
        ticker = TimeEstimator(len(batches))
        for batch in batches:
            x0, x1, x2 = batch
            soft_out, = sess.run([sout, ],
                                  feed_dict={
                                      task.x_list[0]: x0,
                                      task.x_list[1]: x1,
                                      task.x_list[2]: x2,
                                  })
            logit_list.append(soft_out)
            ticker.tick()
        return np.concatenate(logit_list)

    # train_batches, dev_batches = self.load_nli_data(data_loader)
    def idf_explain(enc_payload, explain_tag, forward_run):
        train_batches, dev_batches = get_nli_data(hparam, nli_setting)
        idf_scorer = IdfScorer(train_batches)
        return idf_scorer.explain(enc_payload, explain_tag, forward_run)

    todo_list = [
        ('deletion_seq', explain_by_seq_deletion),
        ('replace_token', explain_by_replace),
        ('term_deletion', explain_by_term_deletion),
        ('term_replace', explain_by_term_replace),
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


def label_predict(hparam, data, model_path) -> List[np.array]:
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
        ticker = TimeEstimator(len(batches))
        for batch in batches:
            x0, x1, x2 = batch
            soft_out, = sess.run([sout, ],
                                  feed_dict={
                                      task.x_list[0]: x0,
                                      task.x_list[1]: x1,
                                      task.x_list[2]: x2,
                                  })
            logit_list.append(soft_out)
            ticker.tick()
        return np.concatenate(logit_list)

    logits = forward_run(enc_payload)
    return logits


