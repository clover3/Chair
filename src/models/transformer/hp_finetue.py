
class HP:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 16  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 512 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 768  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 12
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    intermediate_size = 3072
    type_vocab_size = 2
