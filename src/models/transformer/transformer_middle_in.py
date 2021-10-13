import tensorflow as tf

from models.transformer import bert
from models.transformer.bert_middle_in import BertMiddleIn
from models.transformer.nli_base import ClassificationB
from models.transformer.transformer_cls import transformer_pooled_I


class transformer_middle_in(transformer_pooled_I):
    def __init__(self, hp, voca_size, middle_layer, is_training=True,
                 feed_middle=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        label_ids = tf.placeholder(tf.int64, [None])
        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids
        use_one_hot_embeddings = use_tpu

        if feed_middle:
            self.encoded_embedding_in = tf.placeholder(tf.float32, [None, seq_length, hp.hidden_units])
            self.attention_mask_in = tf.placeholder(tf.float32, [None, seq_length, seq_length])
            model = BertMiddleIn(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
                embeddding_as_input=(self.encoded_embedding_in, self.attention_mask_in),
                middle_layer=middle_layer
            )
        else:
            model = BertMiddleIn(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
                embeddding_as_input=None,
                middle_layer=middle_layer
            )
        self.model = model
        self.encoded_embedding_out = self.model.embedding_output
        self.attention_mask_out = self.model.attention_mask
        self.middle_hidden_vector, self.middle_attention_mask = self.model.middle_output
        pooled_output = self.model.get_pooled_output()
        task = ClassificationB(is_training, hp.hidden_units, 3)
        task.call(pooled_output, label_ids)
        self.loss = task.loss
        self.logits = task.logits
        self.acc = task.acc

    def get_logits(self):
        return self.logits

    def get_input_placeholders(self):
        return self.x_list

    def get_all_encoder_layers(self):
        return self.model.get_all_encoder_layers()

    def get_embedding_output(self):
        return self.model.get_embedding_output()

    def batch2feed_dict(self, batch):
        if len(batch) == 3:
            x0, x1, x2 = batch
            feed_dict = {
                self.x_list[0]: x0,
                self.x_list[1]: x1,
                self.x_list[2]: x2,
            }
        else:
            x0, x1, x2, y = batch
            feed_dict = {
                self.x_list[0]: x0,
                self.x_list[1]: x1,
                self.x_list[2]: x2,
                self.y: y,
            }
        return feed_dict

    def embedding_feed_dict(self, embedding_batch):
        x0, x1, x2, encoded_embedding_in, attention_mask = embedding_batch
        return {
            self.x_list[0]: x0,
            self.x_list[1]: x1,
            self.x_list[2]: x2,
            self.encoded_embedding_in: encoded_embedding_in,
            self.attention_mask_in: attention_mask
        }
