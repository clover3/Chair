import random

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

from tlm.data_gen.base import get_basic_input_feature_as_list_all_ids
from tlm.qtype.BertQType import shift_construct


class ShiftConstructTest(test.TestCase):
    #
    # def _testBitcast(self, x, datatype, shape):
    #     with test_util.use_gpu():
    #         tf_ans = array_ops.bitcast(x, datatype)
    #         out = self.evaluate(tf_ans)
    #         buff_after = memoryview(out).tobytes()
    #         buff_before = memoryview(x).tobytes()
    #         self.assertEqual(buff_before, buff_after)
    #         self.assertEqual(tf_ans.get_shape(), shape)
    #         self.assertEqual(tf_ans.dtype, datatype)

    def test_any(self):
        query_tokens = [2001, 3002, 5005]
        doc_tokens = [random.randint(2000, 10000) for _ in range(30)]
        cls_id = 101
        sep_id = 102
        input_ids = [cls_id] + query_tokens + [sep_id] + doc_tokens + [sep_id]
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_tokens) + 1)
        valid_input_len = len(input_ids)
        max_seq_length = 512
        input_ids, input_mask, segment_ids = get_basic_input_feature_as_list_all_ids(input_ids, segment_ids, max_seq_length)

        self.assertEqual(input_mask[valid_input_len-1], 1)
        self.assertEqual(input_mask[valid_input_len], 0)

        input_ids = tf.expand_dims(input_ids, 0)
        input_mask = tf.expand_dims(input_mask, 0)
        segment_ids = tf.expand_dims(segment_ids, 0)

        hidden_dim = 728
        embedding_table = tf.Variable(np.random.rand(30222, hidden_dim), dtype=tf.float32)
        embedding_output = tf.nn.embedding_lookup(embedding_table, input_ids)
        qtype_ids = [[2222]]
        qtype_embedding = tf.nn.embedding_lookup(embedding_table, qtype_ids)

        shifted_embedding, shifted_input_mask, shifted_segment_ids \
            = shift_construct(embedding_output, qtype_embedding, 1,
                              input_ids, input_mask, segment_ids,
                              sep_id
                              )

        shifted_segment_ids = shifted_segment_ids[0]
        shifted_input_mask = shifted_input_mask[0]
        self.assertNDArrayNear(shifted_embedding[0, 1], qtype_embedding[0, 0], err=1e-8)
        for i in range(0, 1 + len(query_tokens) + 1 + 1):
            self.assertEqual(shifted_segment_ids[i], 0)
            self.assertEqual(shifted_input_mask[i], 1)

        # qtype embedding should shift one
        self.assertEqual(shifted_input_mask[valid_input_len], 1)
        self.assertEqual(shifted_input_mask[valid_input_len+1], 0)
        self.assertEqual(shifted_segment_ids[valid_input_len], 1)


if __name__ == "__main__":
    test.main()
