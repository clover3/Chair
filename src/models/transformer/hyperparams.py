

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

class HPColdStart:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 32  # alias = N
    lr = 5e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 50 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 256 # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 50
    num_heads = 8
    dropout_rate = 0.3
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
    seq_max = 200 # Maximum number of words in a sentence. alias = T.
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
    lr = 1e-4  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    sent_max = 50
    seq_max = 200 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 256 # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.0
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
    batch_size = 60  # alias = N
    lr = 1e-4  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 101 # Maximum number of words in a sentence. alias = T.
    sent_max = 50
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 256  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.

class HPPairFeatureTweet:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 32  # alias = N
    lr = 1e-4  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 50 # Maximum number of words in a sentence. alias = T.
    sent_max = 50
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 256  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    feature_size = 100
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.

class HPPairFeatureTweetFine:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 16 # alias = N
    lr = 3e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 50 # Maximum number of words in a sentence. alias = T.
    sent_max = 50
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 256  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 100
    num_heads = 8
    feature_size = 100
    dropout_rate = 0
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.

class HPStanceConsistency:
    '''Hyperparameters'''
    # data
    # training
    batch_size_sup = 1 # alias = N
    batch_size_aux = 30 # alias = N
    lr = 3e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 50 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    alpha = 0.3
    hidden_units = 256  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 100
    num_heads = 8
    feature_size = 100
    dropout_rate = 0
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.


class HPDocLM:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 24  # alias = N
    lr = 1e-4  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    seq_max = 200 # Maximum number of words in a seqeuence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 256  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 1
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.

