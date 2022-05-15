import os
import sys

import tensorflow as tf
from official.nlp import bert
from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs
from official.nlp.bert.run_classifier import get_predictions_and_labels

from cpath import get_bert_config_path
from taskman_client.wrapper2 import report_run_named
from trainer_v2.arg_flags import flags_parser
from trainer_v2.chair_logging import c_log
from trainer_v2.get_tpu_strategy import get_strategy
from trainer_v2.partial_processing.config_helper import ModelConfig
from trainer_v2.partial_processing.misc_helper import parse_input_files
from trainer_v2.run_config import RunConfigEx, get_run_config_nli_train


@report_run_named("predict_classifier")
def main(args):
    c_log.info("predict_classifier.py")
    input_files = parse_input_files(args.input_files)
    bert_config = bert_configs.BertConfig.from_json_file(get_bert_config_path())
    max_seq_length = 300
    run_config: RunConfigEx = get_run_config_nli_train(args)
    model_config = ModelConfig(bert_config, max_seq_length)

    strategy = get_strategy(args.use_tpu, args.tpu_name)
    def get_model_fn():
        classifier_model, core_model = \
            bert_models.classifier_model(bert_config, model_config.num_classes, max_seq_length)
        return classifier_model, core_model

    eval_input_fn = bert.run_classifier.get_dataset_fn(input_files,
                                                 max_seq_length,
                                                 run_config.batch_size,
                                                 is_training=False)
    with strategy.scope():
        classifier_model, _ = get_model_fn()
        checkpoint = tf.train.Checkpoint(model=classifier_model)
        latest_checkpoint_file = tf.train.latest_checkpoint(args.output_dir)
        c_log.info(f'Checkpoint file {latest_checkpoint_file} found and restoring from checkpoint')
        checkpoint.restore(latest_checkpoint_file).assert_existing_objects_matched()
        c_log.info(f'get_predictions_and_labels')
        preds, _ = get_predictions_and_labels(
            strategy,
            classifier_model,
            eval_input_fn,
            is_regression=False,
            return_probs=True)

        output_predict_file = os.path.join('test_results.tsv')
        with tf.io.gfile.GFile(output_predict_file, 'w') as writer:
          for probabilities in preds:
            output_line = '\t'.join(
                str(class_probability)
                for class_probability in probabilities) + '\n'
            writer.write(output_line)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
