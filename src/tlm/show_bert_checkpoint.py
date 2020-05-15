from misc.show_checkpoint_vars import load_checkpoint_vars


def show_record():
    path = "C:\work\Code\Chair\output\\uncased_L-12_H-768_A-12\\bert_model.ckpt"

    vars = load_checkpoint_vars(path)

    for name, val in vars.items():
        print(name, val.shape)


show_record()