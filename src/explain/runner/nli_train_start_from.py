import sys

from explain.train_nli import train_nil_from
from trainer.model_saver import load_model


def train_nil_resume(model_path, save_dir):
    print("Load model path : ", model_path)
    print("Save dir : ", save_dir)
    def load_fn(sess, model_path):
        return load_model(sess, model_path)

    steps = 73630
    return train_nil_from(save_dir, model_path, load_fn, steps)


if __name__  == "__main__":
    train_nil_resume(sys.argv[1], sys.argv[2])