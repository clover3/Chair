import numpy as np

from tf_util.tf_logging import tf_logging


class NLIPairingTrainConfig:
    vocab_filename = "bert_voca.txt"
    vocab_size = 30522
    seq_length = 300
    max_steps = 100000
    num_gpu = 1
    save_train_payload = False


class HPCommon:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 16  # alias = N
    lr = 2e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    seq_max = 300  # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    hidden_units = 768  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_heads = 12
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    intermediate_size = 3072
    vocab_size = 30522
    type_vocab_size = 2
    num_classes = 3


def find_padding(input_mask):
    return np.where(input_mask == 0)[0][0]


def find_seg2(segment_ids):
    return np.where(segment_ids == 1)[0][0]


def train_fn_factory(sess, loss_tensor, all_losses, train_op, batch2feed_dict, batch, step_i):
    loss_val, all_losses_val, _ = sess.run([loss_tensor, all_losses, train_op,
                                                ],
                                               feed_dict=batch2feed_dict(batch)
                                               )
    n_layer = len(all_losses_val)
    verbose_loss_str = " ".join(["{0}: {1:.2f}".format(i, all_losses_val[i]) for i in range(n_layer)])
    tf_logging.debug("Step {0} train loss={1:.04f} {2}".format(step_i, loss_val, verbose_loss_str))
    return loss_val, 0