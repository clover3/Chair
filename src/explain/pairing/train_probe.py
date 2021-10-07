import os
from collections import defaultdict
from functools import partial
from typing import List

import numpy as np
import tensorflow as tf

from evals.basic_func import get_acc_prec_recall
from explain.nli_common import save_fn_factory
from explain.pairing.match_predictor import ProbeConfigI
from explain.pairing.probe_model import ClsProbeModel
from explain.pairing.probe_train_common import find_padding, find_seg2, train_fn_factory
from misc_lib import average
from tf_util.tf_logging import tf_logging
from trainer.model_saver import setup_summary_writer
from trainer.tf_module import step_runner
from trainer.tf_train_module import init_session


def eval_fn_factory(sess, dev_batches,
                    loss_tensor, per_layer_loss, per_layer_logits_tensor,
                    cls_logits_tensor,
                    global_step_tensor, batch2feed_dict, test_writer):
    per_batch_np_array_d = defaultdict(list)
    all_per_layer_logits: List[List[np.array]] = []
    per_layer_loss_list = []
    for batch in dev_batches:
        input_ids, input_mask, segment_ids, label = batch
        per_batch_np_array_d["input_mask"].append(input_mask)
        per_batch_np_array_d["segment_ids"].append(segment_ids)
        per_batch_np_array_d["label"].append(label)

        loss_val, per_layer_loss_val, per_layer_logits, cls_logits, g_step_val \
            = sess.run([loss_tensor, per_layer_loss, per_layer_logits_tensor, cls_logits_tensor,
                        global_step_tensor],
                       feed_dict=batch2feed_dict(batch)
                       )
        per_batch_np_array_d["loss_val"].append(loss_val)
        per_batch_np_array_d["cls_logits"].append(cls_logits)
        per_layer_loss_list.append(per_layer_loss_val)

        all_per_layer_logits.append(per_layer_logits)

    def get_concat_for(key):
        return np.concatenate(per_batch_np_array_d[key], axis=0)

    all_segment_ids = get_concat_for("segment_ids")
    all_input_mask = get_concat_for("input_mask")
    all_label = get_concat_for("label")
    cls_logits = get_concat_for("cls_logits")

    all_per_layer_loss = np.stack(per_layer_loss_list)
    num_layer = len(per_layer_logits_tensor)
    logits_grouped_by_layer = []
    for layer_no in range(num_layer):
        t = np.concatenate([batch[layer_no] for batch in all_per_layer_logits], axis=0)
        logits_grouped_by_layer.append(t)

    avg_loss = np.average(per_batch_np_array_d["loss_val"])
    summary = tf.Summary()

    tf_logging.info("Step dev step={0} loss={1:.04f}".format(g_step_val, avg_loss))
    summary.value.add(tag='loss', simple_value=avg_loss)
    gold_pred = np.argmax(cls_logits, axis=1)
    gold_pred_exd = np.expand_dims(gold_pred, axis=2)
    for layer_no, logits in enumerate(logits_grouped_by_layer):
        pred = np.argmax(logits, axis=2)

        # score_per_data_point = []
        score_list_d = defaultdict(list)
        for data_point_idx in range(len(gold_pred)):
            # per_data_point = get_acc_prec_recall(pred_binary[data_point_idx], gold_ex_binary[data_point_idx])
            # score_per_data_point.append(per_data_point)
            padding_start = find_padding(all_input_mask[data_point_idx])
            seg2_start = find_seg2(all_segment_ids[data_point_idx])
            assert 1 < seg2_start < padding_start
            condition_list = [(0, len(pred), "all"),
                              (0, padding_start, "non-padding"),
                              (1, seg2_start, "seg1"),
                              (seg2_start, padding_start, "seg2"),
                              (0, 1, "CLS"),
                              (1, padding_start, "non-cls"),
                              ]
            for st, ed, name in condition_list:
                conditioned_score = get_acc_prec_recall(
                    pred[data_point_idx][st:ed],
                    gold_pred_exd[data_point_idx]
                )
                score_list_d[name].append(conditioned_score)

        for condition_name in score_list_d:
            score_list = score_list_d[condition_name]
            scores = {}
            for metric in ["accuracy", "precision", "recall", "f1"]:
                # with tf.name_scope(metric):
                scores[metric] = average([d[metric] for d in score_list])
                tag_name = '{}-{}/Layer{}'.format(condition_name, metric, layer_no)
                summary.value.add(tag=tag_name, simple_value=scores[metric])

            if condition_name == "all":
                tf_logging.debug("Layer {0} acc={1:.02f}, prec={2:.02f} recall={3:.02f} f1={4:.02f}"
                                .format(layer_no, scores['accuracy'], scores['precision'],
                                        scores['recall'], scores['f1']))

        layer_loss = np.mean(all_per_layer_loss[:, layer_no])
        summary.value.add(tag="loss/Layer{}".format(layer_no), simple_value=layer_loss)

    test_writer.add_summary(summary, g_step_val)
    test_writer.flush()

    return avg_loss


def train_probe(bert_hp, train_config, lms_config: ProbeConfigI, save_dir, nli_data,
                init_fn):
    tf_logging.info("train_probe ENTRY")
    train_batches, dev_batches = nli_data

    max_steps = train_config.max_steps
    num_gpu = train_config.num_gpu

    probe_model = ClsProbeModel(bert_hp, lms_config, num_gpu)
    train_cls = probe_model.get_train_op(bert_hp.lr, max_steps)
    global_step = tf.train.get_or_create_global_step()

    run_name = os.path.basename(save_dir)
    tf_logging.info("run name: {}".format(run_name))
    train_writer, test_writer = setup_summary_writer(run_name)

    sess = init_session()
    sess.run(tf.global_variables_initializer())
    init_fn(sess)

    # make explain train_op does not increase global step

    def fetch_global_step():
        step, = sess.run([global_step])
        return step

    train_classification = partial(train_fn_factory,
                                   sess,
                                   probe_model.loss_tensor,
                                   probe_model.per_layer_loss,
                                   train_cls,
                                   probe_model.batch2feed_dict)
    eval_acc = partial(eval_fn_factory, sess, dev_batches[:20],
                       probe_model.loss_tensor,
                       probe_model.per_layer_loss,
                       probe_model.per_layer_logit_tensor,
                       probe_model.logits,
                       global_step,
                       probe_model.batch2feed_dict,
                       test_writer)

    save_fn = partial(save_fn_factory, sess, save_dir, global_step)
    init_step,  = sess.run([global_step])

    def train_fn(batch, step_i):
        loss_val, acc = train_classification(batch, step_i)
        summary = tf.Summary()
        summary.value.add(tag='loss', simple_value=loss_val)
        train_writer.add_summary(summary, fetch_global_step())
        train_writer.flush()
        return loss_val, acc

    def valid_fn():
        eval_acc()
    tf_logging.info("Initialize step to {}".format(init_step))
    tf_logging.info("{} train batches".format(len(train_batches)))
    valid_freq = 100
    save_interval = 600
    loss, _ = step_runner(train_batches, train_fn, init_step,
                          valid_fn, valid_freq,
                          save_fn, save_interval, max_steps)
    return save_fn()


