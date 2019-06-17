from task.transformer_est import  Classification
from models.transformer import bert
import tensorflow as tf

class transformer_mt:
    def __init__(self, hp, voca_size, num_class_list, is_training=True):
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
        input_mask = tf.placeholder(tf.int64, [None, seq_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int64, [None, seq_length], name="segment_ids")

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y1 = tf.placeholder(tf.int64, [None], name="y1")
        self.y2 = tf.placeholder(tf.int64, [None], name="y2")
        self.y = [self.y1, self.y2]
        summary1 = {}
        summary2 = {}
        self.summary_list = [summary1, summary2]

        use_one_hot_embeddings = use_tpu
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        task = Classification(num_class_list[0])
        pred, loss = task.predict(self.model.get_sequence_output(), self.y1, True)
        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss
        self.acc = task.acc
        summary1['loss1'] = tf.summary.scalar('loss', self.loss)
        summary1['acc1'] = tf.summary.scalar('acc', self.acc)

        with tf.variable_scope("cls2"):
            task2 = Classification(num_class_list[1])
            pred, loss = task2.predict(self.model.get_sequence_output(), self.y2, True)
            self.logits2 = task2.logits
            self.sout2 = tf.nn.softmax(self.logits2)
            self.pred2 = pred
            self.loss2 = loss
            self.acc2 = task2.acc
            summary2['loss2'] = tf.summary.scalar('loss2', self.loss2)
            summary2['acc2'] = tf.summary.scalar('acc2', self.acc2)

        self.logit_list = [self.logits, self.logits2]
        self.loss_list = [self.loss, self.loss2]
        self.pred_list = [self.pred, self.pred2]