import os
import sys
from functools import partial

import numpy as np
import tensorflow as tf

from attribution.eval import eval_explain
from data_generator.NLI.nli import get_modified_data_loader
from data_generator.common import get_tokenizer
from data_generator.shared_setting import BertNLI
from explain.nli_common import train_classification_factory, save_fn_factory, valid_fn_factory
from explain.train_ex_core import ExplainTrainerM
from explain.train_nli import get_nli_data
from models.transformer import hyperparams
from models.transformer.nli_base import transformer_nli_pooled
from tf_util.tf_logging import tf_logging, set_level_debug
from trainer.model_saver import setup_summary_writer, load_model_with_blacklist
from trainer.np_modules import get_batches_ex
from trainer.tf_module import step_runner
from trainer.tf_train_module import init_session, get_train_op2


def tag_informative(explain_tag, before_prob, after_prob, action):
    num_tag = np.count_nonzero(action)
    penalty = (num_tag - 3) * 0.1 if num_tag > 3 else 0
    if explain_tag == 'conflict':
        score = (before_prob[2] - before_prob[0]) - (after_prob[2] - after_prob[0])
    elif explain_tag == 'match':
        # Increase of neutral
        score = (before_prob[2] + before_prob[0]) - (after_prob[2] + after_prob[0])
        # ( 1 - before_prob[1] ) - (1 - after_prob[1]) = after_prob[1] - before_prob[1] = increase of neutral
    elif explain_tag == 'mismatch':
        score = before_prob[1] - after_prob[1]
    else:
        assert False
    score = score - penalty
    return score


def train_nli_ex(hparam, nli_setting, save_dir, max_steps, data, data_loader, model_path, modeling_option):
    print("train_nli_ex")
    train_batches, dev_batches = data

    task = transformer_nli_pooled(hparam, nli_setting.vocab_size)
    with tf.variable_scope("optimizer"):
        train_cls = get_train_op2(task.loss, hparam.lr, "adam", max_steps)
    global_step = tf.train.get_or_create_global_step()

    tags = ["conflict", "match", "mismatch"]
    explain_dev_data_list = {tag: data_loader.get_dev_explain(tag) for tag in tags}

    run_name = os.path.basename(save_dir)
    train_writer, test_writer = setup_summary_writer(run_name)

    information_fn_list = list([partial(tag_informative, t) for t in tags])

    def batch2feed_dict(batch):
        if len(batch) == 3:
            x0, x1, x2 = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
            }
        else:
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
        return feed_dict

    explain_trainer = ExplainTrainerM(information_fn_list,
                                      task.logits,
                                      task.model.get_sequence_output(),
                                      len(tags),
                                      batch2feed_dict,
                                      hparam,
                                      modeling_option,
                                      )

    sess = init_session()
    sess.run(tf.global_variables_initializer())
    if model_path is not None:
        load_model_with_blacklist(sess, model_path, ["explain"])

    def eval_tag():
        print("Eval")
        for label_idx in range(3):
            tag = tags[label_idx]
            enc_explain_dev, explain_dev = explain_dev_data_list[tag]
            batches = get_batches_ex(enc_explain_dev, hparam.batch_size, 3)

            ex_logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                ex_logits,  = sess.run([explain_trainer.get_ex_logits(label_idx)],
                                                   feed_dict={
                                                       task.x_list[0]: x0,
                                                       task.x_list[1]: x1,
                                                       task.x_list[2]: x2,
                                                   })
                ex_logit_list.append(ex_logits)
            ex_logit_list = np.concatenate(ex_logit_list)
            assert len(ex_logit_list) == len(explain_dev)
            print(ex_logit_list.shape)
            scores = eval_explain(ex_logit_list, data_loader, tag)

            for metric in scores.keys():
                print("{}\t{}".format(metric, scores[metric]))

            p_at_1, MAP_score = scores["P@1"], scores["MAP"]
            summary = tf.Summary()
            summary.value.add(tag='{}_P@1'.format(tag), simple_value=p_at_1)
            summary.value.add(tag='{}_MAP'.format(tag), simple_value=MAP_score)
            train_writer.add_summary(summary, fetch_global_step())
            train_writer.flush()

    # make explain train_op does not increase global step

    def train_explain(batch, step_i):
        summary = explain_trainer.train_batch(batch, sess)
        train_writer.add_summary(summary, fetch_global_step())

    def fetch_global_step():
        step, = sess.run([global_step])
        return step

    train_classification = partial(train_classification_factory, sess, task.loss, task.acc, train_cls, batch2feed_dict)
    eval_acc = partial(valid_fn_factory, sess, dev_batches[:20], task.loss, task.acc, global_step, batch2feed_dict)

    save_fn = partial(save_fn_factory, sess, save_dir, global_step)
    init_step,  = sess.run([global_step])

    def train_fn(batch, step_i):
        step_before_cls = fetch_global_step()
        loss_val, acc = train_classification(batch, step_i)
        summary = tf.Summary()
        summary.value.add(tag='acc', simple_value=acc)
        summary.value.add(tag='loss', simple_value=loss_val)
        train_writer.add_summary(summary, fetch_global_step())
        train_writer.flush()

        step_after_cls = fetch_global_step()

        assert step_after_cls == step_before_cls + 1
        train_explain(batch, step_i)
        step_after_ex = fetch_global_step()
        assert step_after_cls == step_after_ex
        return loss_val, acc

    def valid_fn():
        eval_acc()
        eval_tag()

    print("Initialize step to {}".format(init_step))
    print("{} train batches".format(len(train_batches)))
    valid_freq = 1000
    save_interval = 5000
    loss, _ = step_runner(train_batches, train_fn, init_step,
                              valid_fn, valid_freq,
                              save_fn, save_interval, max_steps)
    return save_fn()


def train_from(start_model_path, save_dir, modeling_option):
    tf_logging.info("train_from : nli_ex")
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    max_steps = 73630
    set_level_debug()

    tokenizer = get_tokenizer()
    tf_logging.info("Intializing dataloader")
    data_loader = get_modified_data_loader(tokenizer, hp.seq_max, nli_setting.vocab_filename)
    tf_logging.info("loading batches")
    data = get_nli_data(hp, nli_setting)
    train_nli_ex(hp, nli_setting, save_dir, max_steps, data, data_loader, start_model_path, modeling_option)


if __name__  == "__main__":
    train_from(sys.argv[1], sys.argv[2], sys.argv[3])