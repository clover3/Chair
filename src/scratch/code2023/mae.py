import tensorflow as tf


from cpath import output_path
from misc_lib import path_join

Metric = tf.keras.metrics.Metric
class MAELike(Metric):
    def __init__(self, name, **kwargs):
        super(MAELike, self).__init__(name=name, **kwargs)
        self.mae = self.add_weight(name='mae', initializer='zeros',
                                   aggregation=tf.VariableAggregation.MEAN)
        self.count = self.add_weight(name='count', initializer='zeros',
                                     aggregation=tf.VariableAggregation.MEAN)

    def update_state(self, a, b, _sample_weight=None):
        v = tf.reduce_sum(tf.abs(a-b))
        self.mae.assign_add(v)
        self.count.assign_add(1.0)

    def result(self):
        return self.mae / self.count

    def reset_state(self):
        self.mae.assign(0.0)
        self.count.assign(0.0)



def main():
    train_log_dir = path_join(output_path, "train_log_dev")
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        create_file_writer = tf.summary.create_file_writer
        train_summary_writer = create_file_writer(train_log_dir, name="train")
        train_summary_writer.set_as_default()
        name = "mae2"
        metric = MAELike(name)
        step = 0

        def train_step(step):
            y_true = tf.zeros([16,256, 1])
            pred = tf.ones([16, 256, 1])
            metric.update_state(y_true, pred)
            step = tf.cast(step, tf.int64)
            with tf.name_scope("parent/"):
                tf.summary.scalar(name, metric.result(), step=step)

        @tf.function
        def multi_step():
            max_step = 10
            i = 0
            train_step(i)
            for i in tf.range(max_step):
                train_step(i)
            strategy.run(train_step, args=(i, ))

        multi_step()


if __name__ == "__main__":
    main()
