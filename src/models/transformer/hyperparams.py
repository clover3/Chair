

class Hyperparams:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 32  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 128 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 512  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.



class HPDefault:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 32  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 128 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 512  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.


class HPFineTune:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 32  # alias = N
    lr = 3e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 50 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 512  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.


class HPFineTunePair:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 8  # alias = N
    lr = 5e-4  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 101 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 256 # alias = C
    num_blocks = 6  # number of encoder/decoder blocks
    num_epochs = 10
    num_heads = 8
    dropout_rate = 0.4
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.

class HPTweets:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 128  # alias = N
    lr = 1e-4  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 50 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 512  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.

class HPPairTweet:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 64  # alias = N
    lr = 1e-4  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 101 # Maximum number of words in a sentence. alias = T.
    sent_max = 50
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 256  # alias = C
    num_blocks = 6  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.

