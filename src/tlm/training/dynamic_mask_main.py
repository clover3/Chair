import os
import pickle
from functools import partial

import tensorflow as tf

import tlm.dictionary.ssdr_model_fn as ssdr_model_fn
import tlm.model.base as modeling
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, logging
from tlm.dictionary.dict_reader_transformer import DictReaderModel
from tlm.dictionary.sense_selecting_dictionary_reader import SSDR
from tlm.model.base import BertModel
from tlm.model_cnfig import JsonConfig
from tlm.tlm.blc_scorer import brutal_loss_compare
from tlm.tlm.tlm2_network import tlm2, tlm_prefer_hard
from tlm.training.dict_model_fn import model_fn_dict_reader, DictRunConfig, input_fn_builder_dict
from tlm.training.input_fn import input_fn_builder_unmasked, input_fn_builder_masked, input_fn_builder_blc
from tlm.training.lm_model_fn import model_fn_lm, model_fn_target_masking, get_nli_ex_model_segmented
from tlm.training.train_flags import *


class LMTrainConfig:
    def __init__(self,
                 init_checkpoint,
                 learning_rate,
                 num_train_steps,
                 num_warmup_steps,
                 use_tpu,
                 use_one_hot_embeddings,
                 max_predictions_per_seq,
                 gradient_accumulation=1,
                 checkpoint_type="",
                 second_init_checkpoint="",
                 fixed_mask=False,
                 older_logits=False
                 ):
        self.init_checkpoint = init_checkpoint
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.use_tpu = use_tpu
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_predictions_per_seq = max_predictions_per_seq
        self.gradient_accumulation = gradient_accumulation
        self.checkpoint_type = checkpoint_type
        self.second_init_checkpoint = second_init_checkpoint
        self.fixed_mask = fixed_mask

    @classmethod
    def from_flags(cls, flags):
        return LMTrainConfig(
            flags.init_checkpoint,
            flags.learning_rate,
            flags.num_train_steps,
            flags.num_warmup_steps,
            flags.use_tpu,
            flags.use_tpu,
            flags.max_predictions_per_seq,
            flags.gradient_accumulation,
            flags.checkpoint_type,
            flags.target_task_checkpoint,
            flags.fixed_mask,
        )


@report_run
def main(_):
    tf_logging.setLevel(logging.INFO)
    if FLAGS.log_debug:
        tf_logging.setLevel(logging.DEBUG)

    if FLAGS.dbert_config_file:
        FLAGS.model_config_file = FLAGS.dbert_config_file


    tf.io.gfile.makedirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    lm_pretrain(input_files)

    tf_logging.log("Now terminating process")

def lm_pretrain(input_files):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf_logging.info("*** Input Files ***")
    for idx, input_file in enumerate(input_files):
        tf_logging.info("  %s" % input_file)
        if idx > 10 :
          break
    if FLAGS.do_predict:
        seed = 0
    else:
        seed = None

    tf_logging.info("Total of %d files" % len(input_files))

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=False,)
    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_every_n_hours =FLAGS.keep_checkpoint_every_n_hours,
      session_config=config,
      tf_random_seed=seed,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

    TASK_LM = 0
    TASK_TLM = 1
    TASK_DICT_LM = 2
    TASK_DICT_LM_VBATCH = 3

    task = TASK_LM
    if FLAGS.target_lm:
        task = TASK_TLM
    elif FLAGS.dict_lm:
        task = TASK_DICT_LM
    elif FLAGS.dict_lm_vbatch:
        task = TASK_DICT_LM_VBATCH

    train_config = LMTrainConfig.from_flags(FLAGS)
    if task == TASK_LM:
        tf_logging.info("Running LM")
        if FLAGS.fixed_mask:
            input_fn_builder = input_fn_builder_masked
        else:
            input_fn_builder = input_fn_builder_unmasked

        model_fn = model_fn_lm(
            bert_config=bert_config,
            train_config=train_config,
            model_class=BertModel,
        )
    elif task == TASK_TLM:
        tf_logging.info("Running TLM")
        model_config = JsonConfig.from_json_file(FLAGS.model_config_file)

        target_model_config = bert_config
        if model_config.compare_attrib_value_safe("not_twin", True):
            target_model_config = model_config

        input_fn_builder = input_fn_builder_unmasked

        if FLAGS.modeling == "nli_ex":
            priority_model = get_nli_ex_model_segmented
        elif FLAGS.modeling == "tlm2":
            priority_model = partial(tlm2, target_model_config, FLAGS.use_tpu)
        elif FLAGS.modeling == "tlm_hard":
            priority_model = partial(tlm_prefer_hard, target_model_config, FLAGS.use_tpu)
        elif FLAGS.modeling == "BLC":
            priority_model = brutal_loss_compare
            input_fn_builder = input_fn_builder_blc
        else:
            raise Exception()

        model_fn = model_fn_target_masking(
            bert_config=bert_config,
            train_config=train_config,
            target_model_config=model_config,
            model_class=BertModel,
            priority_model=priority_model,
        )
    elif task == TASK_DICT_LM:
        tf_logging.info("Running Dict LM")
        dbert_config = modeling.BertConfig.from_json_file(FLAGS.model_config_file)
        input_fn_builder = input_fn_builder_dict
        model_fn = model_fn_dict_reader(
            bert_config=bert_config,
            dbert_config=dbert_config ,
            train_config=train_config,
            logging=tf_logging,
            model_class=DictReaderModel,
            dict_run_config=DictRunConfig.from_flags(FLAGS),
        )
    elif task == TASK_DICT_LM_VBATCH:
        tf_logging.info("Running Dict LM with virtual batch_size")
        ssdr_config = JsonConfig.from_json_file(FLAGS.model_config_file)

        if FLAGS.modeling == "mockup":
            input_fn_builder = input_fn_builder_unmasked
            model_fn = ssdr_model_fn.model_fn_apr_lm(
                bert_config=bert_config,
                ssdr_config=ssdr_config,
                train_config=train_config,
                dict_run_config=DictRunConfig.from_flags(FLAGS),
            )
        elif FLAGS.modeling == "debug":
            tf_logging.info("Running Debugging")
            input_fn_builder = ssdr_model_fn.input_fn_builder
#            input_fn_builder = input_fn_builder_unmasked

            model_fn = ssdr_model_fn.model_fn_apr_debug(
                bert_config=bert_config,
                ssdr_config=ssdr_config ,
                train_config=train_config,
                logging=tf_logging,
                model_name="APR",
                dict_run_config=DictRunConfig.from_flags(FLAGS),
            )
        elif FLAGS.modeling == "debug2":
            tf_logging.info("Running Debugging2")
            input_fn_builder = ssdr_model_fn.input_fn_builder
            #            input_fn_builder = input_fn_builder_unmasked

            model_fn = ssdr_model_fn.model_fn_apr_debug(
                bert_config=bert_config,
                ssdr_config=ssdr_config,
                train_config=train_config,
                logging=tf_logging,
                model_name="BERT",
                dict_run_config=DictRunConfig.from_flags(FLAGS),
            )
        else:
            input_fn_builder = ssdr_model_fn.input_fn_builder
            model_fn = ssdr_model_fn.model_fn_dict_reader(
                bert_config=bert_config,
                ssdr_config=ssdr_config ,
                train_config=train_config,
                logging=tf_logging,
                model_class=SSDR,
                dict_run_config=DictRunConfig.from_flags(FLAGS),
            )
    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.eval_batch_size,
    )

    if FLAGS.do_train:
        tf_logging.info("***** Running training *****")
        tf_logging.info("  Batch size = %d", FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            flags=FLAGS,
            is_training=True)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        tf_logging.info("***** Running evaluation *****")
        tf_logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            flags=FLAGS,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
          tf_logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf_logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        return result

    if FLAGS.do_predict:
        tf_logging.info("***** Running prediction *****")
        tf_logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        predict_input_fn = input_fn_builder(
            input_files=input_files,
            flags=FLAGS,
            is_training=False,
        )
        result = estimator.predict(input_fn=predict_input_fn, yield_single_examples=False)
        tf_logging.info("***** Pickling.. *****")
        pickle.dump(list(result), open(FLAGS.out_file, "wb"))



if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
