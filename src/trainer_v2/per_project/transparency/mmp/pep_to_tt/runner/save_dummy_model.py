import tensorflow as tf
import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.eval_loop import tf_run_eval
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel, PairwiseEvaler
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.per_project.transparency.mmp.pep_to_tt.dataset_builder import read_pep_tt_dataset
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig, \
    PEP_TT_Model_Single, define_inputs_single, bm25_like
from trainer_v2.train_util.arg_flags import flags_parser


class DummyTT(ModelV2IF):
    def __init__(self, model_config: PEP_TT_ModelConfig):
        self.model_config = model_config
        super(ModelV2IF, self).__init__()

    def build_model(self, _):
        self.max_seq_length = self.model_config.max_seq_length
        self.build_pairwise_train_network()

    def build_pairwise_train_network(self):
        max_seq_length = self.model_config.max_seq_length
        inputs_d = define_inputs_single(max_seq_length)
        # [batch_size, dim]

        score_d = {}
        for role in ["pos", "neg"]:
            input_ids = inputs_d[f"{role}_input_ids"]
            segment_ids = inputs_d[f"{role}_segment_ids"]
            norm_add_factor = inputs_d[f"{role}_norm_add_factor"]
            multiplier = inputs_d[f"{role}_multiplier"]
            value_score = inputs_d[f"{role}_value_score"]
            probs = tf.expand_dims(tf.zeros_like(value_score), axis=1)
            total_score = bm25_like(probs, multiplier, norm_add_factor, value_score)
            score_d[role] = total_score

        score_stack = tf.stack([score_d["pos"], score_d["neg"]], axis=1)
        losses = tf.maximum(1 - (score_d["pos"] - score_d["neg"]), 0)
        loss = tf.reduce_mean(losses)
        outputs = score_stack, loss
        model = tf.keras.Model(inputs=inputs_d, outputs=outputs, name="bert_model")
        self.model: tf.keras.Model = model

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, model_path):
        pass


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    model_config = PEP_TT_ModelConfig()
    task_model = DummyTT(model_config)
    task_model.build_model(None)
    task_model.model.save(run_config.train_config.model_save_path)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
