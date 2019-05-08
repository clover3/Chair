from task.transformer_est import  Classification
from models.transformer import bert
import tensorflow as tf

class transformer_weight:
    def __init__(self, hp, voca_size, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )
        seq_length = hp.seq_max
        use_tpu = False

        input_ids = tf.placeholder(tf.int64, [None, seq_length], name="input_ids")
        input_mask = tf.placeholder(tf.float32, [None, seq_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int64, [None, seq_length], name="segment_ids")

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = tf.placeholder(tf.int64, [None], name="y")
        self.summary = {}

        use_one_hot_embeddings = use_tpu
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        task = Classification(3)
        pred, loss = task.predict(self.model.get_sequence_output(), self.y, True)
        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss
        self.acc = task.acc
        self.summary['loss1'] = tf.summary.scalar('loss', self.loss)
        self.summary['acc1'] = tf.summary.scalar('acc', self.acc)

