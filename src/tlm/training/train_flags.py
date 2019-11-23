from absl import flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
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
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_string(
    "modeling", "multiblock", "Which model to use"
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

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_every_n_hours", 1,
                     "Number of hours between each checkpoint to be saved.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_integer("max_pred_steps", 100, "Maximum number of prediction steps.")

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

flags.DEFINE_bool("log_debug", False, ".")

flags.DEFINE_bool("fixed_mask", False, "Whether to fixed_mask.")

flags.DEFINE_bool("repeat_data", True, "Whether to repeat data.")

flags.DEFINE_bool("is_bert_checkpoint", True, "init_checkpoint is from BERT")

flags.DEFINE_integer("num_classes", 3, "Number of classes (in case of classification task.")

flags.DEFINE_integer("gradient_accumulation", 1, "How many batch to accumulate for gradient update.")

