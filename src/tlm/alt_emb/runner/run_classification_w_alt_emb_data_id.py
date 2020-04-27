
from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model.embedding_replacer import EmbeddingReplacer2
from tlm.model_cnfig import JsonConfig
from tlm.training.classification_model_fn import model_fn_classification
from tlm.training.flags_wrapper import get_input_files_from_flags
from tlm.training.input_fn import input_fn_builder_alt_emb_data_id_classification
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Classification with alternative embedding, + data_id")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)

    is_training = FLAGS.do_train
    input_files = get_input_files_from_flags(FLAGS)

    input_fn = input_fn_builder_alt_emb_data_id_classification(input_files, FLAGS, is_training)

    def model_constructor(config, is_training, input_ids,
                          input_mask, token_type_ids, use_one_hot_embeddings, features):
        return EmbeddingReplacer2(config, is_training, input_ids,
                          input_mask, token_type_ids, use_one_hot_embeddings, features)
    special_flags = FLAGS.special_flags.split(",")
    special_flags.append("feed_features")
    special_flags.append("ask_tvar")

    model_fn = model_fn_classification(config, train_config, model_constructor, special_flags)
    run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
