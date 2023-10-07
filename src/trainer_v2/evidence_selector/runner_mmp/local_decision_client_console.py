from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import TwoSegConcatEncoderQD
from port_info import LOCAL_DECISION_PORT
from trainer_v2.custom_loop.definitions import ModelConfig512_1, ModelConfig512_2
from utils.xml_rpc_helper import ServerProxyEx


def replace_mask(tokens):
    i = 0
    out_tokens = []
    while i < len(tokens):
        if i+2 < len(tokens) and tokens[i] == "[" and tokens[i+1] == "mask" and tokens[i+2] == "]":
            out_tokens.append("[MASK]")
            i += 3
        else:
            out_tokens.append(tokens[i])
            i += 1
    return out_tokens


def main():
    model_config = ModelConfig512_2()
    tokenizer = get_tokenizer()
    encoder = TwoSegConcatEncoderQD(tokenizer, model_config.max_seq_length)

    server_addr = "localhost"
    proxy = ServerProxyEx(server_addr, LOCAL_DECISION_PORT)
    def tokenize(text):
        return replace_mask(tokenizer.tokenize(text))

    tokenizer = get_tokenizer()
    while True:
        sent1 = input("(Partial) Document: ")
        sent2_1 = input("(Partial) Query1: ")
        sent2_2 = input("(Partial) Query2: ")
        d_tokens = tokenize(sent1)
        q_first = tokenize(sent2_1)
        q_second = tokenize(sent2_2)
        print(d_tokens)
        print(q_first)
        print(q_second)
        triplet = encoder.two_seg_concat_core(d_tokens, q_first, q_second)
        input_ids, input_mask, segment_ids = triplet
        x = input_ids, segment_ids

        print(x)
        print("Sending...")
        res_list = proxy.send([x])
        result = res_list[0]
        print(result)
        l_decisions, g_decision = result
        print((sent1, sent2_1, sent2_2))
        print(l_decisions[0], l_decisions[1])
        print(g_decision)


if __name__ == "__main__":
    main()
