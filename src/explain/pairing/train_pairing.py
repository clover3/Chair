import os
from collections import defaultdict
from functools import partial
from typing import List

import numpy as np
import tensorflow as tf

from evals.basic_func import get_acc_prec_recall
from explain.nli_common import save_fn_factory
from explain.pairing.lms_model import LMSModel
from explain.pairing.match_predictor import ProbeConfigI
from explain.pairing.probe_train_common import find_padding, find_seg2, train_fn_factory
from misc_lib import average
from tf_util.tf_logging import tf_logging
from trainer.model_saver import setup_summary_writer
from trainer.tf_module import step_runner
from trainer.tf_train_module import init_session


def eval_fn_factory(sess, dev_batches,
                    loss_tensor, per_layer_loss, ex_scores_tensor, per_layer_logits_tensor,
                    global_step_tensor, batch2feed_dict, test_writer):
    loss_list = []
    all_ex_scores: List[np.array] = []
    all_per_layer_logits: List[List[np.array]] = []
    input_mask_list = []
    per_layer_loss_list = []
    segment_ids_list = []
    label_list = []
    for batch in dev_batches:
        input_ids, input_mask, segment_ids, label = batch
        input_mask_list.append(input_mask)
        segment_ids_list.append(segment_ids)
        label_list.append(label)
        loss_val, per_layer_loss_val, ex_scores, per_layer_logits, g_step_val \
            = sess.run([loss_tensor, per_layer_loss, ex_scores_tensor, per_layer_logits_tensor, global_step_tensor],
                       feed_dict=batch2feed_dict(batch)
                       )
        loss_list.append(loss_val)
        all_ex_scores.append(ex_scores)
        per_layer_loss_list.append(per_layer_loss_val)
        all_per_layer_logits.append(per_layer_logits)

    all_segment_ids = np.concatenate(segment_ids_list, axis=0)
    all_input_mask = np.concatenate(input_mask_list, axis=0)
    all_ex_scores_np = np.concatenate(all_ex_scores, axis=0)
    all_per_layer_loss = np.stack(per_layer_loss_list)
    all_label = np.concatenate(label_list, axis=0)
    num_layer = len(per_layer_logits_tensor)
    logits_grouped_by_layer = []
    for layer_no in range(num_layer):
        t = np.concatenate([batch[layer_no] for batch in all_per_layer_logits], axis=0)
        logits_grouped_by_layer.append(t)

    avg_loss = np.average(loss_list)
    summary = tf.Summary()

    tf_logging.info("Step dev step={0} loss={1:.04f}".format(g_step_val, avg_loss))
    summary.value.add(tag='loss', simple_value=avg_loss)
    gold_ex_binary = all_ex_scores_np > 0.5
    for layer_no, logits in enumerate(logits_grouped_by_layer):
        pred_binary = np.less(logits[:, :, 0], logits[:, :, 1])

        # score_per_data_point = []
        score_list_d = defaultdict(list)
        for data_point_idx in range(len(gold_ex_binary)):
            # per_data_point = get_acc_prec_recall(pred_binary[data_point_idx], gold_ex_binary[data_point_idx])
            # score_per_data_point.append(per_data_point)
            padding_start = find_padding(all_input_mask[data_point_idx])
            seg2_start = find_seg2(all_segment_ids[data_point_idx])
            assert 1 < seg2_start < padding_start
            for st, ed, name in [(0, len(pred_binary), "all"),
                                 (0, padding_start, "non-padding"),
                                 (1, seg2_start, "seg1"),
                                 (seg2_start, padding_start, "seg2")]:
                conditioned_score = get_acc_prec_recall(
                    pred_binary[data_point_idx][st:ed],
                    gold_ex_binary[data_point_idx][st:ed]
                )
                score_list_d[name].append(conditioned_score)

            if all_label[data_point_idx] == 1:
                st = seg2_start
                ed = padding_start
                conditioned_score = get_acc_prec_recall(
                    pred_binary[data_point_idx][st:ed],
                    gold_ex_binary[data_point_idx][st:ed]
                )
                score_list_d["neutral"].append(conditioned_score)

        for condition_name in score_list_d:
            score_list = score_list_d[condition_name]
            scores = {}
            for metric in ["accuracy", "precision", "recall", "f1"]:
                # with tf.name_scope(metric):
                scores[metric] = average([d[metric] for d in score_list])
                tag_name = '{}-{}/Layer{}'.format(condition_name, metric, layer_no)
                summary.value.add(tag=tag_name, simple_value=scores[metric])

            if condition_name == "all":
                tf_logging.info("Layer {0} acc={1:.02f}, prec={2:.02f} recall={3:.02f} f1={4:.02f}"
                                .format(layer_no, scores['accuracy'], scores['precision'],
                                        scores['recall'], scores['f1']))

        layer_loss = np.mean(all_per_layer_loss[:, layer_no])
        summary.value.add(tag="loss/Layer{}".format(layer_no), simple_value=layer_loss)

    test_writer.add_summary(summary, g_step_val)
    test_writer.flush()

    return avg_loss


def train_LMS(bert_hp, train_config, lms_config: ProbeConfigI, save_dir, nli_data, modeling_option, init_fn):
    tf_logging.info("train_pairing ENTRY")
    train_batches, dev_batches = nli_data

    max_steps = train_config.max_steps
    num_gpu = train_config.num_gpu

    lms_model = LMSModel(modeling_option, bert_hp, lms_config, num_gpu)
    train_cls = lms_model.get_train_op(bert_hp.lr, max_steps)
    global_step = tf.train.get_or_create_global_step()

    run_name = os.path.basename(save_dir)
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
                                   lms_model.loss_tensor,
                                   lms_model.per_layer_loss,
                                   train_cls,
                                   lms_model.batch2feed_dict)
    eval_acc = partial(eval_fn_factory, sess, dev_batches[:20],
                       lms_model.loss_tensor,
                       lms_model.per_layer_loss,
                       lms_model.ex_score_tensor,
                       lms_model.per_layer_logit_tensor,
                       global_step,
                       lms_model.batch2feed_dict,
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
    save_interval = 300
    loss, _ = step_runner(train_batches, train_fn, init_step,
                              valid_fn, valid_freq,
                              save_fn, save_interval, max_steps)
    return save_fn()


