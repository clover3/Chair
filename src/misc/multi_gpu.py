import numpy as np
import tensorflow as tf

from models.transformer.nli_base import ClassificationB
from models.transformer.optimization import AdamWeightDecayOptimizer
from tf_v2_support import placeholder
from trainer.tf_train_module import init_session

batch_size = 8


class FF:
    def __init__(self, input_tensor):
        self.logits = tf.layers.dense(input_tensor, 2, name="cls_dense")
        self.task = ClassificationB(True, 10, 2)
        self.task.call(input_tensor, tf.ones([batch_size], tf.int32))

def run():

    all_loss = 0
    tower_grads = []

    models = []
    for gpu_idx in range(2):
        with tf.device("/gpu:{}".format(gpu_idx)):
            with tf.variable_scope("vars", reuse=gpu_idx > 0):
                input_x = placeholder(tf.float32, [None, 10])
                model = FF(input_x)
                models.append(model)
                tf.get_variable_scope().reuse_variables()
                all_loss += model.task.loss

    tvars = tf.trainable_variables()
    for t in tvars:
        print(t.name)

    for gpu_idx in range(2):
        grads = tf.gradients(model.task.loss, tvars)
        print(grads)
        # Keep track of the gradients across all towers.
        tower_grads.append(grads)

    avg_grads = []
    for t_idx, _ in enumerate(tvars):
        g1 = tower_grads[0][0]
        g2 = tower_grads[1][1]

        g_avg = (g1 + g2) / 2 if g1 is not None else None
        avg_grads.append(g_avg)

    global_step = tf.Variable(0, name='global_step')
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=0.001,
        weight_decay_rate=0.02,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    train_cls = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    #train_cls = get_train_op2(all_loss, 0.001, "adam", 10000)
    sess = init_session()
    sess.run(tf.global_variables_initializer())

    def train_classification():
        loss_val, _ = sess.run([model.task.loss, train_cls],
                               feed_dict={input_x:np.ones([batch_size, 10])}
                               )
        print(loss_val)


    for i in range(100000):
        print("Train")
        train_classification()


if __name__ == "__main__":
    run()