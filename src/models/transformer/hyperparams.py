

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


class  HPDefault:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 32  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 200 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 512  # alias = C
    num_blocks = 8  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1

    sinusoid = False  # If True, use sinusoid. If false, positional embedding.


class HPNLI2:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 32  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 200 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 512  # alias = C
    num_blocks = 8  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    intermediate_size = hidden_units * 2
    type_vocab_size = 16
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.


class HPBert:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 16  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 200 # Maximum number of words in a sentence. alias = T.
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


class HPSENLI2:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 16  # alias = N
    lr = 2e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    seq_max = 200 # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    hidden_units = 768  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_heads = 12
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    intermediate_size = 3072
    type_vocab_size = 2
    g_val = 0.5


class HPSENLI3:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 16  # alias = N
    lr = 2e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    seq_max = 300 # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    hidden_units = 768  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_heads = 12
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    intermediate_size = 3072
    type_vocab_size = 2
    g_val = 0.5


class HPSENLI:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 16  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 200 # Maximum number of words in a sentence. alias = T.
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
    g_val = 0.5


class HPCausal:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 16  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 256 # Maximum number of words in a sentence. alias = T.
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


class HPUbuntu:
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
    num_epochs = 1
    num_heads = 12
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    intermediate_size = 3072
    type_vocab_size = 2
    alpha = 1


class HPAdhoc:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 16  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 200 # Maximum number of words in a sentence. alias = T.
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
    alpha = 0.1



class HPFAD:
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
    alpha = 0.1


class HPMerger_BM25:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 32  # alias = N
    lr = 1e-4  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 100 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 32  # alias = C
    num_blocks = 4  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    type_vocab_size = 2
    intermediate_size = 128
    alpha = 1



class HPMerger:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 32  # alias = N
    lr = 1e-4  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 40 # number of instance to merge
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 32  # alias = C
    num_blocks = 4  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    type_vocab_size = 2
    intermediate_size = 128
    alpha = 1




class HPQL:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 512  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 200 # Maximum number of words in a sentence. alias = T.
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
    alpha = 0.5
    query_seq_len = 20



class HPMscore:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 16  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 200 # Maximum number of words in a sentence. alias = T.
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
    alpha = 0.1


class HPNLI:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 32  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 128 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 256  # alias = C
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
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 50 # Maximum number of words in a sentence. alias = T.
    #lm_seq_len = 512
    # Feel free to increase this if you are ambitious.
    hidden_units = 256 # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.3
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.


class HPTiny:
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
    hidden_units = 64# alias = C
    num_blocks = 5  # number of encoder/decoder blocks
    num_epochs = 50
    num_heads = 4
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
    num_epochs = 4
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.


class HPFineTunePair:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 8  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
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




class HPCIE:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 4  # alias = N
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
    alpha = 0.5
    query_seq_len = 20




class HPCNN:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 64  # alias = N
    lr = 1e-3 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 1024 # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    reg_lambda = 0.1
    dropout_rate = 0.5
    embedding_size= 300




class HPUKPVector:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 12  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 200 # Maximum number of words in a sentence. alias = T.
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
    fixed_v = 10
    num_v = 2
    use_reorder= False





class HPCRS:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 4  # alias = N
    lr = 1e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    #seq_max = 140  # Maximum number of words in a sentence. alias = T.
    seq_max = 256 # Maximum number of words in a sentence. alias = T.
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
    alpha = 0.5
    query_seq_len = 20
