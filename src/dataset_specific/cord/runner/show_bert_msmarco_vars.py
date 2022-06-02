from misc.show_checkpoint_vars import load_checkpoint_vars


def show_record():
    path = "C:\work\Code\Chair\output\model\BERT_Base_trained_on_MSMARCO\model.ckpt-100000"
    path = "C:\work\Code\Chair\output\model\msmarco_2\msmarco_2"
    path = "C:\work\Code\Chair\output\model\msmarco_2\msmarco_2"

    vars = load_checkpoint_vars(path)

    for name, val in vars.items():
        print(name, val.shape)


show_record()
