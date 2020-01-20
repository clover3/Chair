import os

import math
import numpy as np
import scipy
from scipy.special import softmax

from cpath import get_bert_full_path, output_path, data_path
from misc_lib import average
from models.transformer.bert_common_v2 import gelu
from tf_v2_support import disable_eager_execution
from tlm.model.base import embedding_lookup, BertConfig, embedding_postprocessor, \
    create_attention_mask_from_input_mask, reshape_to_matrix, dense, create_initializer, layer_norm
from tlm.tlm.runner.analyze_param import load_param
from trainer.model_saver import load_v2_to_v2
from trainer.tf_train_module_v2 import init_session
from visualize.html_visual import HtmlVisualizer, Cell


def get_attention_param(all_param, layer_no):
    common_prefix = "bert/encoder/layer_{}/attention/self/".format(layer_no)

    W_q = all_param[common_prefix + "query/kernel"]
    b_q = all_param[common_prefix + "query/bias"]
    W_k = all_param[common_prefix + "key/kernel"]
    b_k = all_param[common_prefix + "key/bias"]
    return W_q, b_q, W_k, b_k

hidden_size= 768
num_head = 12
head_size = int(hidden_size / num_head)
num_attention_heads = 12
size_per_head = head_size

def get_embedding_score(attention_param, source_emb, target_emb):
    W_q, b_q, W_k, b_k = attention_param

    query = np.matmul(source_emb, W_q) + b_q
    query = np.reshape(query, [num_head, head_size])
    key = np.matmul(target_emb, W_k) + b_k
    key = np.reshape(key, [num_head, head_size])
    scores = np.sum(np.multiply(query, key), axis=-1)
    scores = scores * (1.0 / math.sqrt(float(head_size)))
    return scores

def apply_layer_norm(y, gamma, beta):
    mean = np.mean(y, axis=-1)
    variance = np.var(y, axis=-1)

    normalized_y = (y - mean) / np.sqrt(variance + 1e-5)
    y_out = normalized_y * gamma + beta
    return y_out


import tensorflow as tf


class BertLike:
    def __init__(self):
        config = BertConfig.from_json_file(os.path.join(data_path, "bert_config.json"))
        self.attention_probs_list = []

        input_ids = tf.constant([[101] + [100] * 511])
        token_type_ids = tf.constant([[0] * 512])
        input_mask = tf.constant([[1]*512])
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)
        initializer = create_initializer(config.initializer_range)

        scope = None
        with tf.compat.v1.variable_scope(scope, default_name="bert"):
            with tf.compat.v1.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=False)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)
            prev_output = reshape_to_matrix(self.embedding_output)
            with tf.compat.v1.variable_scope("encoder"):

                for layer_idx in range(12):
                    with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                        layer_input = prev_output

                        with tf.compat.v1.variable_scope("attention"):
                            attention_heads = []
                            with tf.compat.v1.variable_scope("self"):
                                attention_head = self.attention_fn(layer_input)
                                attention_heads.append(attention_head)

                            attention_output = None
                            if len(attention_heads) == 1:
                                attention_output = attention_heads[0]
                            else:
                                # In the case where we have other sequences, we just concatenate
                                # them to the self-attention head before the projection.
                                attention_output = tf.concat(attention_heads, axis=-1)

                            # Run a linear projection of `hidden_size` then add a residual
                            # with `layer_input`.
                            with tf.compat.v1.variable_scope("output"):
                                attention_output = dense(hidden_size, initializer)(attention_output)
                                attention_output = layer_norm(attention_output + layer_input)

                        # The activation is only applied to the "intermediate" hidden layer.
                        with tf.compat.v1.variable_scope("intermediate"):
                            intermediate_output = dense(config.intermediate_size, initializer,
                                                        activation=gelu)(attention_output)

                        # Down-project back to `hidden_size` then add the residual.
                        with tf.compat.v1.variable_scope("output"):
                            layer_output = dense(hidden_size, initializer)(intermediate_output)
                            layer_output = layer_norm(layer_output + attention_output)
                            prev_output = layer_output

    def attention_fn(self, tensor):
        # `query_layer` = [B*F, N*H]
        query_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=None,
            name="query")(tensor)

        # `key_layer` = [B*T, N*H]
        key_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=None,
            name="key")(tensor)

        # `value_layer` = [B*T, N*H]
        value_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            name="value")(tensor)

        def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                                 seq_length, width):
            output_tensor = tf.reshape(
                input_tensor, [batch_size, seq_length, num_attention_heads, width])

            output_tensor = tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])
            return output_tensor

        # `value_layer` = [B*T, N*H]
        # `query_layer` = [B, N, F, H]
        query_layer = transpose_for_scores(query_layer, 1,
                                           num_attention_heads, 512,
                                           size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(key_layer, 1, num_attention_heads,
                                         512, size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(size_per_head)))
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        self.attention_probs_list.append(attention_probs)
    
    
        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)
    
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
    
        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
                value_layer,
                [1, 512, num_attention_heads, size_per_head])
    
        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(a=value_layer, perm=[0, 2, 1, 3])
    
        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)
    
        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])
    
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(
                context_layer,
                [1, 512, num_attention_heads * size_per_head])

        return context_layer

def load_bert_like():
    disable_eager_execution()
    model = BertLike()
    sess = init_session()
    #sess.run(tf.compat.v1.global_variables_initializer())
    load_v2_to_v2(sess, get_bert_full_path() , False)

    attention_prob_list, = sess.run([model.attention_probs_list])
    html = HtmlVisualizer("position.html")

    for layer_no, attention_prob in enumerate(attention_prob_list):
        html.write_headline("Layer {}".format(layer_no))
        acc_dict = {}

        zero_scores = [list() for _ in range(12)]

        for loc in range(2, 40, 2):
            print("Source : ", loc)
            for target_loc in range(20):
                offset = target_loc - loc

                print(offset, end= " ")
                for head_idx in range(num_head):
                    key = offset, head_idx
                    if key not in acc_dict:
                        acc_dict[key] = []
                    e = attention_prob[0, head_idx, loc, target_loc]
                    if target_loc != 0:
                        acc_dict[key].append(e)
                    else:
                        zero_scores[head_idx].append(e)
                    print("{0:.2f}".format(e*100) , end=" ")
                print()

        rows = [[Cell("Loc")] + [Cell("Head{}".format(i)) for i in range(12)]]
        for offset in range(-7, +7):
            print(offset, end=" ")
            scores = []
            for head_idx in range(12):
                key = offset, head_idx

                try:
                    elems = acc_dict[key]
                    if len(elems) < 3:
                        raise KeyError

                    avg = average(elems)
                    scores.append(avg)
                    print("{0:.2f}".format(avg*100) , end=" ")
                except KeyError:
                    print("SKIP")
            print()
            rows.append([Cell(offset)] +[Cell(float(v*100), v * 1000) for v in scores])
        html.write_table(rows)

        html.write_paragraph("Attention to first token")
        zero_scores = [average(l) for l in zero_scores]
        rows = [[Cell("   ")]+[Cell("Head{}".format(i)) for i in range(12)],
                [Cell("   ")]+[Cell(float(v*100), v*1000) for v in zero_scores]]
        html.write_table(rows)

def all_job():
    bert_parameter = load_param(get_bert_full_path())

    # TODO : Attention effect of position embedding only
    # TODO : Cosine similarity of position embeddings,
    # TODO : Query-Value scoring of position embeddings,
    # TODO : Query-Value scoring of pos_emb + OOV_emb
    # TODO : Cosine similarity of hidden dimension in each layer
    key_pos_embedding = "bert/embeddings/position_embeddings"

    pos_embedding = bert_parameter[key_pos_embedding]
    word_embedding = bert_parameter["bert/embeddings/word_embeddings"]
    seg_embedding = bert_parameter["bert/embeddings/token_type_embeddings"]
    layer_norm_gamma = bert_parameter["bert/embeddings/LayerNorm/gamma"]
    layer_norm_beta = bert_parameter["bert/embeddings/LayerNorm/beta"]
    oov_emb = word_embedding[100]
    cls_emb = word_embedding[101]
    seg1_emb = seg_embedding[0]
    print(pos_embedding.shape)

    def get_embedding(location):
        if location == 0:
            emb = pos_embedding[location] + oov_emb + seg1_emb
        else:
            emb = pos_embedding[location] + oov_emb + seg1_emb

        return apply_layer_norm(emb, layer_norm_gamma, layer_norm_beta)



    p = os.path.join(output_path, "hv_lm.pickle")
    #hv_lm = pickle.load(open(p, "rb"))

    layer_0_param = get_attention_param(bert_parameter, 0)
    for source_location in [0, 5]:#, 20, 40]:
        source_emb = get_embedding(source_location)
        scores = []
        tar_loc_list = range(512)
        # for offset in [-5, -2, -1, 0, 1, 2, 5]:
        #     target_location = source_location + offset
        #     if target_location >=0 and target_location < 512:
        #         tar_loc_list.append(target_location)
        #

        for target_location in tar_loc_list:
            target_emb = get_embedding(target_location)
            score = get_embedding_score(layer_0_param, source_emb, target_emb)
            scores.append(score)

        scores = np.array(scores)
        print(scores.shape)
        scores = scipy.special.softmax(scores, axis=1)

        print("From {}".format(source_location))
        for idx, score in enumerate(scores[:30]):
            print(idx, end=" ")
            for e in score:
                print(e, end=" ")
            print()




if __name__ == "__main__":
    load_bert_like()
#    all_job()
