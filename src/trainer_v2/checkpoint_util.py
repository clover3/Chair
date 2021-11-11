import tensorflow as tf


def load_weights(bert_encoder, checkpoint_path):
    checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
    ret = checkpoint.read(checkpoint_path)
    ret.assert_existing_objects_matched()
    print(ret.__dict__)
    ret.assert_consumed()

