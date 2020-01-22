import tensorflow as tf

from models.transformer.bert_common_v2 import get_shape_list
from models.transformer.optimization_v2 import create_optimizer
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.training.input_fn import input_fn_builder_unmasked
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


class OomReportingHook(tf.estimator.SessionRunHook):
    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs(fetches=[],  # no extra fetches
                      options=tf.compat.v1.RunOptions(
                      report_tensor_allocations_upon_oom=True))



def model_fn_builder(modeling, batch_size):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        next_sentence_labels = features["next_sentence_labels"]

        initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.02)

        vocab_size = 40000
        embedding_size = 512
        embedding_table = tf.compat.v1.get_variable(
            name="embedding",
            shape=[vocab_size, embedding_size],
            initializer=initializer)

        input_shape = get_shape_list(input_ids)
        flat_input_ids = tf.reshape(input_ids, [-1])
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
        output = tf.reshape(output, input_shape + [embedding_size])

        t_list = []
        n_of_t = 10
        for j in range(n_of_t):
            t_list.append(output+j)



        dense = tf.keras.layers.Dense(embedding_size, kernel_initializer=initializer, name="MDense")
        if modeling == "B":
            dense_list = []
            for j in range(n_of_t):
                dense_list.append(tf.keras.layers.Dense(embedding_size,
                                                        kernel_initializer=initializer, name="MDense_{}".format(j)))


        for i in range(20):
            if modeling == "A":
                t = tf.stack(t_list, 0)
                with tf.compat.v1.variable_scope("scope_A", reuse=i > 0):
                    t = dense(t)
                t = tf.nn.dropout(t, rate=0.5)

                t_0 = 0
                for j in range(1, n_of_t):
                    t_0 += t[j]

                new_t_list = [t_0]
                for j in range(1, n_of_t):
                    new_t_list.append(t[j])

                t_list = new_t_list
            else:
                with tf.compat.v1.variable_scope("scope_B", reuse=i > 0):
                    temp_t = []
                    for j in range(n_of_t):
                        t = dense_list[j](t_list[j])
                        t = tf.nn.dropout(t, rate=0.5)
                        temp_t.append(t)

                    t_0 = 0
                    for j in range(1, n_of_t):
                        t_0 += temp_t[j]

                    new_t_list = [t_0]
                    for j in range(1, n_of_t):
                        new_t_list.append(temp_t[j])

                    t_list = new_t_list
                
        t = t_list[0]
        total_loss = tf.reduce_mean(t)
        for t in tf.compat.v1.trainable_variables():
            print(t)
        train_op = create_optimizer(total_loss, 1e-4, 1000, 1000, True)

        scaffold_fn = None
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                train_op=train_op,
                loss=total_loss,
                training_hooks=[OomReportingHook()],
                scaffold_fn=scaffold_fn)


        return output_spec

    return model_fn



@report_run
def main(_):
    is_training = FLAGS.do_train

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    input_fn = input_fn_builder_unmasked(input_files, FLAGS, is_training)
    model_fn = model_fn_builder(FLAGS.modeling, FLAGS.train_batch_size)

    run_estimator(model_fn, input_fn)

if __name__ == "__main__":
    tf.compat.v1.app.run()
