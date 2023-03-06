import re

target = """
distilbert.embeddings.LayerNorm.bias
distilbert.embeddings.LayerNorm.weight
distilbert.embeddings.position_embeddings.weight
distilbert.embeddings.word_embeddings.weight
distilbert.transformer.layer.0.attention.k_lin.bias
distilbert.transformer.layer.0.attention.k_lin.weight
distilbert.transformer.layer.0.attention.out_lin.bias
distilbert.transformer.layer.0.attention.out_lin.weight
distilbert.transformer.layer.0.attention.q_lin.bias
distilbert.transformer.layer.0.attention.q_lin.weight
distilbert.transformer.layer.0.attention.v_lin.bias
distilbert.transformer.layer.0.attention.v_lin.weight
distilbert.transformer.layer.0.ffn.lin1.bias
distilbert.transformer.layer.0.ffn.lin1.weight
distilbert.transformer.layer.0.ffn.lin2.bias
distilbert.transformer.layer.0.ffn.lin2.weight
distilbert.transformer.layer.0.output_layer_norm.bias
distilbert.transformer.layer.0.output_layer_norm.weight
distilbert.transformer.layer.0.sa_layer_norm.bias
distilbert.transformer.layer.0.sa_layer_norm.weight
distilbert.transformer.layer.1.attention.k_lin.bias
distilbert.transformer.layer.1.attention.k_lin.weight
distilbert.transformer.layer.1.attention.out_lin.bias
distilbert.transformer.layer.1.attention.out_lin.weight
distilbert.transformer.layer.1.attention.q_lin.bias
distilbert.transformer.layer.1.attention.q_lin.weight
distilbert.transformer.layer.1.attention.v_lin.bias
distilbert.transformer.layer.1.attention.v_lin.weight
distilbert.transformer.layer.1.ffn.lin1.bias
distilbert.transformer.layer.1.ffn.lin1.weight
distilbert.transformer.layer.1.ffn.lin2.bias
distilbert.transformer.layer.1.ffn.lin2.weight
distilbert.transformer.layer.1.output_layer_norm.bias
distilbert.transformer.layer.1.output_layer_norm.weight
distilbert.transformer.layer.1.sa_layer_norm.bias
distilbert.transformer.layer.1.sa_layer_norm.weight
distilbert.transformer.layer.2.attention.k_lin.bias
distilbert.transformer.layer.2.attention.k_lin.weight
distilbert.transformer.layer.2.attention.out_lin.bias
distilbert.transformer.layer.2.attention.out_lin.weight
distilbert.transformer.layer.2.attention.q_lin.bias
distilbert.transformer.layer.2.attention.q_lin.weight
distilbert.transformer.layer.2.attention.v_lin.bias
distilbert.transformer.layer.2.attention.v_lin.weight
distilbert.transformer.layer.2.ffn.lin1.bias
distilbert.transformer.layer.2.ffn.lin1.weight
distilbert.transformer.layer.2.ffn.lin2.bias
distilbert.transformer.layer.2.ffn.lin2.weight
distilbert.transformer.layer.2.output_layer_norm.bias
distilbert.transformer.layer.2.output_layer_norm.weight
distilbert.transformer.layer.2.sa_layer_norm.bias
distilbert.transformer.layer.2.sa_layer_norm.weight
distilbert.transformer.layer.3.attention.k_lin.bias
distilbert.transformer.layer.3.attention.k_lin.weight
distilbert.transformer.layer.3.attention.out_lin.bias
distilbert.transformer.layer.3.attention.out_lin.weight
distilbert.transformer.layer.3.attention.q_lin.bias
distilbert.transformer.layer.3.attention.q_lin.weight
distilbert.transformer.layer.3.attention.v_lin.bias
distilbert.transformer.layer.3.attention.v_lin.weight
distilbert.transformer.layer.3.ffn.lin1.bias
distilbert.transformer.layer.3.ffn.lin1.weight
distilbert.transformer.layer.3.ffn.lin2.bias
distilbert.transformer.layer.3.ffn.lin2.weight
distilbert.transformer.layer.3.output_layer_norm.bias
distilbert.transformer.layer.3.output_layer_norm.weight
distilbert.transformer.layer.3.sa_layer_norm.bias
distilbert.transformer.layer.3.sa_layer_norm.weight
distilbert.transformer.layer.4.attention.k_lin.bias
distilbert.transformer.layer.4.attention.k_lin.weight
distilbert.transformer.layer.4.attention.out_lin.bias
distilbert.transformer.layer.4.attention.out_lin.weight
distilbert.transformer.layer.4.attention.q_lin.bias
distilbert.transformer.layer.4.attention.q_lin.weight
distilbert.transformer.layer.4.attention.v_lin.bias
distilbert.transformer.layer.4.attention.v_lin.weight
distilbert.transformer.layer.4.ffn.lin1.bias
distilbert.transformer.layer.4.ffn.lin1.weight
distilbert.transformer.layer.4.ffn.lin2.bias
distilbert.transformer.layer.4.ffn.lin2.weight
distilbert.transformer.layer.4.output_layer_norm.bias
distilbert.transformer.layer.4.output_layer_norm.weight
distilbert.transformer.layer.4.sa_layer_norm.bias
distilbert.transformer.layer.4.sa_layer_norm.weight
distilbert.transformer.layer.5.attention.k_lin.bias
distilbert.transformer.layer.5.attention.k_lin.weight
distilbert.transformer.layer.5.attention.out_lin.bias
distilbert.transformer.layer.5.attention.out_lin.weight
distilbert.transformer.layer.5.attention.q_lin.bias
distilbert.transformer.layer.5.attention.q_lin.weight
distilbert.transformer.layer.5.attention.v_lin.bias
distilbert.transformer.layer.5.attention.v_lin.weight
distilbert.transformer.layer.5.ffn.lin1.bias
distilbert.transformer.layer.5.ffn.lin1.weight
distilbert.transformer.layer.5.ffn.lin2.bias
distilbert.transformer.layer.5.ffn.lin2.weight
distilbert.transformer.layer.5.output_layer_norm.bias
distilbert.transformer.layer.5.output_layer_norm.weight
distilbert.transformer.layer.5.sa_layer_norm.bias
distilbert.transformer.layer.5.sa_layer_norm.weight
vocab_layer_norm.bias
vocab_layer_norm.weight
vocab_projector.bias
vocab_projector.weight
vocab_transform.bias
vocab_transform.weight
"""

source = """bert/embeddings/word_embeddings/embeddings:0
bert/embeddings/token_type_embeddings/embeddings:0
bert/embeddings/position_embeddings/embeddings:0
bert/embeddings/LayerNorm/gamma:0
bert/embeddings/LayerNorm/beta:0
bert/encoder/layer_0/attention/self/query/kernel:0
bert/encoder/layer_0/attention/self/query/bias:0
bert/encoder/layer_0/attention/self/key/kernel:0
bert/encoder/layer_0/attention/self/key/bias:0
bert/encoder/layer_0/attention/self/value/kernel:0
bert/encoder/layer_0/attention/self/value/bias:0
bert/encoder/layer_0/attention/output/dense/kernel:0
bert/encoder/layer_0/attention/output/dense/bias:0
bert/encoder/layer_0/attention/output/LayerNorm/gamma:0
bert/encoder/layer_0/attention/output/LayerNorm/beta:0
bert/encoder/layer_0/intermediate/kernel:0
bert/encoder/layer_0/intermediate/bias:0
bert/encoder/layer_0/output/dense/kernel:0
bert/encoder/layer_0/output/dense/bias:0
bert/encoder/layer_0/output/LayerNorm/gamma:0
bert/encoder/layer_0/output/LayerNorm/beta:0
bert/encoder/layer_1/attention/self/query/kernel:0
bert/encoder/layer_1/attention/self/query/bias:0
bert/encoder/layer_1/attention/self/key/kernel:0
bert/encoder/layer_1/attention/self/key/bias:0
bert/encoder/layer_1/attention/self/value/kernel:0
bert/encoder/layer_1/attention/self/value/bias:0
bert/encoder/layer_1/attention/output/dense/kernel:0
bert/encoder/layer_1/attention/output/dense/bias:0
bert/encoder/layer_1/attention/output/LayerNorm/gamma:0
bert/encoder/layer_1/attention/output/LayerNorm/beta:0
bert/encoder/layer_1/intermediate/kernel:0
bert/encoder/layer_1/intermediate/bias:0
bert/encoder/layer_1/output/dense/kernel:0
bert/encoder/layer_1/output/dense/bias:0
bert/encoder/layer_1/output/LayerNorm/gamma:0
bert/encoder/layer_1/output/LayerNorm/beta:0
bert/encoder/layer_2/attention/self/query/kernel:0
bert/encoder/layer_2/attention/self/query/bias:0
bert/encoder/layer_2/attention/self/key/kernel:0
bert/encoder/layer_2/attention/self/key/bias:0
bert/encoder/layer_2/attention/self/value/kernel:0
bert/encoder/layer_2/attention/self/value/bias:0
bert/encoder/layer_2/attention/output/dense/kernel:0
bert/encoder/layer_2/attention/output/dense/bias:0
bert/encoder/layer_2/attention/output/LayerNorm/gamma:0
bert/encoder/layer_2/attention/output/LayerNorm/beta:0
bert/encoder/layer_2/intermediate/kernel:0
bert/encoder/layer_2/intermediate/bias:0
bert/encoder/layer_2/output/dense/kernel:0
bert/encoder/layer_2/output/dense/bias:0
bert/encoder/layer_2/output/LayerNorm/gamma:0
bert/encoder/layer_2/output/LayerNorm/beta:0
bert/encoder/layer_3/attention/self/query/kernel:0
bert/encoder/layer_3/attention/self/query/bias:0
bert/encoder/layer_3/attention/self/key/kernel:0
bert/encoder/layer_3/attention/self/key/bias:0
bert/encoder/layer_3/attention/self/value/kernel:0
bert/encoder/layer_3/attention/self/value/bias:0
bert/encoder/layer_3/attention/output/dense/kernel:0
bert/encoder/layer_3/attention/output/dense/bias:0
bert/encoder/layer_3/attention/output/LayerNorm/gamma:0
bert/encoder/layer_3/attention/output/LayerNorm/beta:0
bert/encoder/layer_3/intermediate/kernel:0
bert/encoder/layer_3/intermediate/bias:0
bert/encoder/layer_3/output/dense/kernel:0
bert/encoder/layer_3/output/dense/bias:0
bert/encoder/layer_3/output/LayerNorm/gamma:0
bert/encoder/layer_3/output/LayerNorm/beta:0
bert/encoder/layer_4/attention/self/query/kernel:0
bert/encoder/layer_4/attention/self/query/bias:0
bert/encoder/layer_4/attention/self/key/kernel:0
bert/encoder/layer_4/attention/self/key/bias:0
bert/encoder/layer_4/attention/self/value/kernel:0
bert/encoder/layer_4/attention/self/value/bias:0
bert/encoder/layer_4/attention/output/dense/kernel:0
bert/encoder/layer_4/attention/output/dense/bias:0
bert/encoder/layer_4/attention/output/LayerNorm/gamma:0
bert/encoder/layer_4/attention/output/LayerNorm/beta:0
bert/encoder/layer_4/intermediate/kernel:0
bert/encoder/layer_4/intermediate/bias:0
bert/encoder/layer_4/output/dense/kernel:0
bert/encoder/layer_4/output/dense/bias:0
bert/encoder/layer_4/output/LayerNorm/gamma:0
bert/encoder/layer_4/output/LayerNorm/beta:0
bert/encoder/layer_5/attention/self/query/kernel:0
bert/encoder/layer_5/attention/self/query/bias:0
bert/encoder/layer_5/attention/self/key/kernel:0
bert/encoder/layer_5/attention/self/key/bias:0
bert/encoder/layer_5/attention/self/value/kernel:0
bert/encoder/layer_5/attention/self/value/bias:0
bert/encoder/layer_5/attention/output/dense/kernel:0
bert/encoder/layer_5/attention/output/dense/bias:0
bert/encoder/layer_5/attention/output/LayerNorm/gamma:0
bert/encoder/layer_5/attention/output/LayerNorm/beta:0
bert/encoder/layer_5/intermediate/kernel:0
bert/encoder/layer_5/intermediate/bias:0
bert/encoder/layer_5/output/dense/kernel:0
bert/encoder/layer_5/output/dense/bias:0
bert/encoder/layer_5/output/LayerNorm/gamma:0
bert/encoder/layer_5/output/LayerNorm/beta:0
"""

target_keys = target.split()
source_keys = source.split()


def transform(source_key):
    conversion_reg_list = [
        (r"bert", "distilbert"),
        (r"encoder", "transformer"),
        (r"layer_(\d)", r"layer.\1"),
        (r"attention/output/LayerNorm", "sa_layer_norm"),
        (r"attention/self/key", "attention.k_lin"),
        (r"attention/self/query", "attention.q_lin"),
        (r"attention/self/value", "attention.v_lin"),
        (r"attention/output/dense", "attention.out_lin"),
        (r"intermediate", "ffn.lin1"),
        (r"(\d)/output/dense", r"\1/ffn.lin2"),
        (r"(\d)/output/LayerNorm", r"\1/output_layer_norm"),
        ("_embeddings/embeddings:0", "_embeddings.weight"),
        ("gamma:0", "weight"),
        ("beta:0", "bias"),
        ("kernel:0", "weight"),
        ("bias:0", "bias")
    ]
    # bert -> distilbert
    # encoder -> transformer
    # layer_5 -> layer.5
    # attention/output/LayerNorm -> sa_layer_norm
    # attention/self/key/ -> attention.k_lin
    # attention/self/query/ -> attention.q_lin
    # attention/self/value/ -> attention.v_lin
    # attention/output/dense -> attention.out_lin

    # intermediate -> ffn.lin1
    # output/dense -> ffn.lin2
    # output/LayerNorm -> output_layer_norm


    # vocab_layer_norm
    # word_embeddings/embeddings:0 -> word_embeddings.weight
    # position_embeddings/embeddings:0 -> position_embeddings.weight
    # token_type_embeddings/embeddings:0  -> ?????
    # LayerNorm/gamma:0 -> LayerNorm.weight
    # LayerNorm/beta:0 -> LayerNorm.bias
    # kernel:0 -> weight
    # bias:0 -> bias

    s = source_key
    for pattern, replacee in conversion_reg_list:
        s = re.sub(pattern, replacee, s)

    s = s.replace("/", ".")
    return s

def do_test():
    collision_d = set()
    for key in source_keys:
        key_to_search = transform(key)
        if key_to_search in collision_d:
            raise KeyError()
        collision_d.add(key_to_search)
        if not key_to_search in target_keys:
            print("{} -> {} ".format(key, key_to_search))
            print("Found = {}".format(key_to_search in target_keys))

