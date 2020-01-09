import sys
from functools import partial

import numpy as np
import tensorflow as tf

from attribution.eval import eval_explain
from cache import save_to_pickle, load_cache
from data_generator.NLI import nli
from data_generator.common import get_tokenizer
from data_generator.shared_setting import NLI
from explain.train_ex import ExplainTrainerM
from misc_lib import average
from models.transformer import hyperparams
from models.transformer.nli_base import transformer_nli_pooled
from tf_util.tf_logging import tf_logging, logging
from trainer.model_saver import save_model, load_bert_v2, setup_summary_writer
from trainer.np_modules import get_batches_ex
from trainer.tf_module import step_runner, get_nli_batches_from_data_loader
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


def train_nli_ex(hparam, nli_setting, run_name, data, data_loader, model_path):
    print("train_nli_ex")
    train_batches, dev_batches = data

    task = transformer_nli_pooled(hparam, nli_setting.vocab_size)
    with tf.variable_scope("optimizer"):
        train_cls = get_train_op2(task.loss, hparam.lr, 75000)

    tags = ["conflict", "match", "mismatch"]
    explain_dev_data_list = {tag: data_loader.get_dev_explain(tag) for tag in tags}
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
                    )

    sess = init_session()
    sess.run(tf.global_variables_initializer())
    if model_path is not None:
        load_bert_v2(sess, model_path)

    g_step = 0
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
            train_writer.add_summary(summary, g_step)
            train_writer.flush()



    def train_explain(batch, step_i):
        summary = explain_trainer.train_batch(batch, sess)
        train_writer.add_summary(summary, g_step)

    def train_classification(batch, step_i):
        loss_val, acc, _ = sess.run([task.loss, task.acc, train_cls,
                                                   ],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
        tf_logging.debug("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        return loss_val, acc

    def train_fn(batch, step_i):
        loss_val, acc = train_classification(batch, step_i)
        train_explain(batch, step_i)
        nonlocal g_step
        g_step += 1
        return loss_val, acc

    def eval_acc():
        loss_list = []
        acc_list = []
        for batch in dev_batches[:10]:
            loss_val, acc = sess.run([task.loss, task.acc],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )
            loss_list.append(loss_val)
            acc_list.append(acc)
        tf_logging.info("Step dev loss={0:.04f} acc={1:.03f}".format(average(loss_list), average(acc_list)))

    def valid_fn():
        eval_acc()
        eval_tag()

    def save_fn():
        global_step = tf.train.get_or_create_global_step()
        return save_model(sess, run_name, global_step)

    valid_freq = 1000
    save_interval = 1000
    print("Total of {} train batches".format(len(train_batches)))
    steps = int(len(train_batches) * 0.5)
    loss, _ = step_runner(train_batches, train_fn,
                          valid_fn, valid_freq,
                          save_fn, save_interval,
                          steps=steps)


def train_nil_from_v2_checkpoint(run_name, model_path):
    tf_logging.info("train_nil_from_v2_checkpoint")
    hp = hyperparams.HPSENLI2()
    hp.seq_max = 300
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    tf_logging.info("Intializing dataloader")
    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    tokenizer = get_tokenizer()
    CLS_ID = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    data_loader.CLS_ID = CLS_ID
    data_loader.SEP_ID = SEP_ID
    tf_logging.setLevel(logging.INFO)
    #model_path = get_model_path("nli_batch16", 'model.ckpt-61358')
    tf_logging.info("loading batches")
    data = load_cache("train_nil_from_v2_checkpoint")
    if data is None:
        data = get_nli_batches_from_data_loader(data_loader, hp.batch_size)
        save_to_pickle(data, "train_nil_from_v2_checkpoint")
    train_nli_ex(hp, nli_setting, run_name,
                 data, data_loader, model_path)


if __name__  == "__main__":
    train_nil_from_v2_checkpoint(sys.argv[1], sys.argv[2])