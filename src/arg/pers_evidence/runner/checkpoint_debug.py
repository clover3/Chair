from misc.show_checkpoint_vars import load_checkpoint_vars


def main():
    ck500 = "/tmp/model.ckpt-500"
    qck0 = "/tmp/model.ckpt-0"

    var_d1 = load_checkpoint_vars(ck500)
    var_d2 = load_checkpoint_vars(qck0)

    var_name1 = "bert/encoder/layer_9/output/dense_29/bias/adam_v"
    var_name2 = "SCOPE2/bert/encoder/layer_9/output/dense_66/bias/adam_v"

    print(var_d1[var_name1])
    print(var_d2[var_name2])


if __name__ == "__main__":
    main()