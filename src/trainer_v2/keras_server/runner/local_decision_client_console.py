from data_generator.tokenizer_wo_tf import get_tokenizer
from port_info import LOCAL_DECISION_PORT
from trainer_v2.custom_loop.definitions import ModelConfig600_3
from trainer_v2.custom_loop.per_task.nli_ts_helper import EncoderType
from trainer_v2.custom_loop.per_task.nli_ts_util import get_two_seg_concat_encoder
from utils.xml_rpc_helper import ServerProxyEx


def main():

    model_config = ModelConfig600_3()
    encode_fn: EncoderType = get_two_seg_concat_encoder(model_config.max_seq_length)
    server_addr = "localhost"
    proxy = ServerProxyEx(server_addr, LOCAL_DECISION_PORT)

    tokenizer = get_tokenizer()
    while True:
        sent1 = input("(Partial) Premise: ")
        sent2_1 = input("(Partial) Hypothesis1: ")
        sent2_2 = input("(Partial) Hypothesis2: ")
        p_tokens = tokenizer.tokenize(sent1)
        h_first = tokenizer.tokenize(sent2_1)
        h_second = tokenizer.tokenize(sent2_2)
        x = encode_fn(p_tokens, h_first, h_second)
        print(x)
        res_list = proxy.send([x])
        print(res_list)
        result = res_list[0]
        print(result)
        l_decisions, g_decision = result
        print((sent1, sent2_1, sent2_2))
        print(l_decisions[0], l_decisions[1])
        print(g_decision)


if __name__ == "__main__":
    main()
