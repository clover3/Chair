
import sys

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.mmp.one_q_term_modeling import get_dataset, get_model, ScoringLayer2
from trainer_v2.train_util.arg_flags import flags_parser
import tensorflow as tf

def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    voca_size = 700

    def build_dataset(input_files, is_for_training):
        return get_dataset(
            input_files, voca_size, run_config)

    run_name = str(run_config.common_run_config.run_name)
    c_log.info("Run name: %s", run_name)

    train_dataset = build_dataset(run_config.dataset_config.train_files_path, True)
    if run_config.dataset_config.eval_files_path:
        eval_dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
    else:
        eval_dataset = None

    train_itr = iter(train_dataset)
    cnt = 0
    c_log.info("Building model")
    network = ScoringLayer2(voca_size)
    for batch in train_itr:
        x, y = batch
        target_idx = 6
        # print("x", x)
        print(tf.experimental.numpy.nonzero(x['x1'][target_idx]))
        print(tf.experimental.numpy.nonzero(x['x2'][target_idx]))
        logits1 = network(x['x1'])
        logits2 = network(x['x2'])
        score1 = x['score1']
        score2 = x['score2']
        print("logits", logits1, logits2)
        print("score1[target_idx]", score1[target_idx], score2[target_idx])
        score1_f = score1 + logits1
        score2_f = score2 + logits2
        pred = score1_f - score2_f
        # pred = tf.expand_dims(score1_f - score2_f, axis=2)
        # print("Pred", pred)
        print("pred[target_idx]", pred[target_idx])
        # print("y", y)
        hinge_loss = tf.keras.losses.Hinge(
        )

        manual_loss = tf.maximum(1 - tf.cast(y, tf.float32) * pred, 0)
        loss = hinge_loss(y, pred)
        print("manual_loss[target_idx]", manual_loss[target_idx])
        # print("manual_loss", manual_loss)
        print("sum(manual_loss)", tf.reduce_sum(manual_loss))
        print("loss", loss)

        cnt += 1
        if cnt > 2:
            break
    learning_rate = 1e-2
    # score1_f = score1 + logits1
    # score2_f = score2 + logits2
    # # inputs = [feature_ids1, feature_values1, score1, feature_ids2, feature_values2, score2]
    # inputs = [x1, score1, x2, score2]
    # pred = score1_f - score2_f
    #
    #

if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


