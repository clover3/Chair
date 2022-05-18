from typing import List

from trainer_v2.kera_debug.dev_name_mapping_resource import v2_ckpt_variables_raw, name_in_ts_model, \
    v1_ckpt_variables_raw


def is_var_restore(variable_name):
    # "model/layer_with_weights-10/_attention_output_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"
    if not variable_name.endswith(".ATTRIBUTES/VARIABLE_VALUE"):
        return False
    if not variable_name.startswith("model"):
        return False

    return True


def parse_step1(variable_name) -> List[str]:
    tokens = variable_name.split("/")
    return tokens


def normalize_ckpt2(s):
    postfix = ".ATTRIBUTES/VARIABLE_VALUE"
    if s.endswith(postfix):
        s = s[:-len(postfix)]


def normalize_mem_var_inner(prefix, s):
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    one_or_two = s.split(":")
    s2 = one_or_two[0]  # drop postfix ":0"
    assert s2.startswith(prefix)
    post_text = s2[len(prefix):]
    post_tokens = post_text.split("/")
    # assert bert_encoder_module == "bert_encoder_module"
    # assert tokens[1] == "encoder1"
    # post_tokens = tokens[2:]

    if post_tokens[1] == "embeddings":
        emb_type = {
            "type_embeddings": "token_type_embeddings",
            "position_embedding": "position_embeddings",
            "word_embeddings": "word_embeddings"
        }[post_tokens[0]]
        return ["bert", "embeddings", emb_type] # e.g., bert/position_embedding/embeddings
    if post_tokens[0] == "embeddings" and post_tokens[1] == "layer_norm":
        gamma_or_beta = post_tokens[2]
        assert gamma_or_beta in ["gamma", "beta"]
        return ["bert", "embeddings", "LayerNorm", gamma_or_beta]

    if post_tokens[0] == "transformer":
        layer_name = post_tokens[1]
        role_name = post_tokens[2]
        if role_name == 'self_attention':
            if post_tokens[3] in ["query", "key", "value"]:
                kernel_or_bias = post_tokens[4]
                assert kernel_or_bias in ["kernel", "bias"]
                return ["bert", "encoder", layer_name, "attention", "self", post_tokens[3], kernel_or_bias]
            elif post_tokens[3] == 'attention_output':
                kernel_or_bias = post_tokens[4]
                assert kernel_or_bias in ["kernel", "bias"]
                return ["bert", "encoder", layer_name, "attention", "output", "dense", kernel_or_bias]
            else:
                assert False
        elif role_name == "self_attention_layer_norm":
            gamma_or_beta = post_tokens[3]
            assert gamma_or_beta in ["gamma", "beta"]
            return ["bert", "encoder", layer_name, "attention", "output", "LayerNorm", gamma_or_beta]
        elif role_name in ["intermediate", "output"]:
            kernel_or_bias = post_tokens[3]
            assert kernel_or_bias in ["kernel", "bias"]
            return ["bert", "encoder", layer_name, role_name, "dense", kernel_or_bias]
        elif role_name in ["output_layer_norm"]:
            gamma_or_beta = post_tokens[3]
            assert gamma_or_beta in ["gamma", "beta"]
            return ["bert", "encoder", layer_name, "output", "LayerNorm", gamma_or_beta]
        else:
            assert False
    elif post_tokens[0] == "pooler_transform":
        kernel_or_bias = post_tokens[1]
        assert kernel_or_bias in ["kernel", "bias"]
        return ["bert", "pooler", "dense", kernel_or_bias]
    else:
        assert False


def main():
    v2_ckpt_variables = v2_ckpt_variables_raw.strip().split("\n")
    v1_ckpt_variables = v1_ckpt_variables_raw.strip().split("\n")

    ckpt_1 = filter(is_var_restore, v2_ckpt_variables)
    ckpt_2: List[List[str]] = list(map(parse_step1, ckpt_1))
    # print(ckpt_2)

    # new_var_names = module_variables_raw.strip().split("\n")
    mem_var_names = name_in_ts_model.strip().split("\n")
    print(f"Memory has {len(mem_var_names)} variables")

    prefix = "bert_encoder_module/encoder1"
    mem_var_mapping = {k: normalize_mem_var_inner(prefix, k) for k in mem_var_names}
    for name in mem_var_names:
        norm_name = "/".join(mem_var_mapping[name])

        if norm_name not in v1_ckpt_variables:
            print(name)
            print("Not found: ", norm_name)

    # "bert_encoder_module/encoder1/transformer/layer_0/self_attention/attention_output/kernel:0"
    # Goal

    # print(ckpt_variables_list)
    # module_variables = module_variables_raw.split("\n")
    # return NotImplemented


if __name__ == "__main__":
    main()