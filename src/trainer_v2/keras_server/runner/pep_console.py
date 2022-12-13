from trainer_v2.keras_server.name_short_cuts import get_pep_client, NLIPredictorSig


def get_input_clean_tab(msg):
    s = input(msg)
    return " ".join(s.split())


def main():
    forward_fn_raw: NLIPredictorSig = get_pep_client()
    while True:
        sent1 = get_input_clean_tab("(Partial) Premise: ")
        sent2_1 = get_input_clean_tab("(Partial) Hypothesis: ")

        res_list = forward_fn_raw([(sent1, sent2_1)])
        print("Sending...")
        result = res_list[0]
        print((sent1, sent2_1))
        print(result)


if __name__ == "__main__":
    main()
