import tensorflow as tf

from tlm.model.base import BertModel


class FreezeEmbedding(BertModel):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 scope=None):
        self.exclude = ["bert/embeddings/word_embeddings:0"]
        super(FreezeEmbedding, self).__init__(config,
                 is_training,
                 input_ids,
                 input_mask,
                 token_type_ids,
                 use_one_hot_embeddings,
                 scope)

    def get_trainable_vars(self):
        r =[]
        for v in tf.compat.v1.trainable_variables():
            if v.name in self.exclude:
                print("Skip: ", v.name)
            else:
                print("Trainable:", v.name)
                r.append(v)

        return r
