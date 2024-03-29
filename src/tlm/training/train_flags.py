from absl import flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "model_config_file", None,
    "The config json file corresponding to the second part of the transformer model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "dbert_config_file", None,
    "The config json file corresponding to the second part of the transformer model. "
    "This specifies the model architecture.")


flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "target_task_checkpoint", None,
    "Checkpoint for target task.")

## Other parameters
flags.DEFINE_string(
    "third_init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_d_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")


flags.DEFINE_integer(
    "max_def_length", 256,
    "max length for dictionary entry")


flags.DEFINE_integer(
    "max_loc_length", 10,
    "max length for locations of words")

flags.DEFINE_integer(
    "max_word_length", 0,
    "max length lookup words")


flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_string(
    "modeling", None, "Which model to use"
)

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run predicition .")

flags.DEFINE_string(
    "train_op", "LM", "what to predict"
)

flags.DEFINE_string(
    "prediction_op", "", "what to predict"
)

flags.DEFINE_string(
    "out_file", "result.pickle", "name for the output file"
)
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("learning_rate2", 5e-4, "The initial learning rate for second part of model.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_train_epochs", 2, "If applicable, it precedes the train_steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_every_n_hours", 1,
                     "Number of hours between each checkpoint to be saved.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "Maximum number of checkpoint to keep.")

flags.DEFINE_integer("cycle_length", 250, "number of parallel files read")

flags.DEFINE_integer("block_length", 1, "The number of consecutive elements to pull from "
                                        "an input Dataset before advancing to the next input Dataset.")

flags.DEFINE_integer("buffer_size", 1000000, "size of shuffling buffer")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", None, "Maximum number of eval steps.")

flags.DEFINE_integer("max_pred_steps", 0, "Maximum number of prediction steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("use_target_pos_emb", False, "Whether to use use_target_pos_emb.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


flags.DEFINE_bool("target_lm", False, "Whether to run target_lm as training.")

flags.DEFINE_bool("dict_lm", False, "Whether to run dict_lm as training.")

flags.DEFINE_bool("dict_lm_vbatch", False, "Whether to run dict_lm as training.")

flags.DEFINE_bool("log_debug", False, ".")

flags.DEFINE_bool("fixed_mask", False, "Whether to fixed_mask.")

flags.DEFINE_bool("not_use_next_sentence", False, "not_use_next_sentence.")

flags.DEFINE_bool("use_d_segment_ids", False, "Whether to use d_segment_ids.")

flags.DEFINE_bool("pool_dict_output", False, "pool_dict_output.")

flags.DEFINE_bool("repeat_data", True, "Whether to repeat data.")

flags.DEFINE_bool("is_bert_checkpoint", True, "init_checkpoint is from BERT")

flags.DEFINE_string("special_flags", "", "special_flags, separated by comma ,")

flags.DEFINE_integer("num_classes", 3, "Number of classes (in case of classification task.")

flags.DEFINE_integer("gradient_accumulation", 1, "How many batch to accumulate for gradient update.")

flags.DEFINE_string("checkpoint_type", "", "Checkpoint type")

flags.DEFINE_string("task_completion_mark", "", "make this file if the task is completed. Do not start the job if this file exists")

flags.DEFINE_integer("inner_batch_size", 60, "Number of classes (in case of classification task.")

flags.DEFINE_integer("def_per_batch", 180, "Number of classes (in case of classification task.")

flags.DEFINE_integer("job_id", -1, "Job number assigned by executor")

flags.DEFINE_integer("random_seed", None, "Number of classes (in case of classification task.")

flags.DEFINE_bool("use_ab_mapping_mask", False, "Whether to use ab_mapping_mask.")

flags.DEFINE_bool("use_cache", False, "Whether to use cached data.")

flags.DEFINE_bool("initialize_to_predict", False, ".")

flags.DEFINE_bool("no_lr_decay", False, ".")

flags.DEFINE_bool("use_old_logits", True,
                  "Whether to use older version of logistic regression for classification_model_fn.")

flags.DEFINE_string(
    "run_name", None,
    ".")

flags.DEFINE_string("report_field", None, "Report this field of the dictionary")
flags.DEFINE_string("report_condition", None, "Report this condition ")


flags.DEFINE_integer("max_query_len", 0, "")
flags.DEFINE_integer("max_doc_len", 0, "")


def model_config_flag_checking():
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
