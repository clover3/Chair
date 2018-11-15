from models.transformer.transformer import *
from trainer.experiment import Experiment
from models.transformer.hyperparams import Hyperparams

def lm_train():
    hp = Hyperparams()
    e = Experiment(hp)
    data_getter = NotImplementedError

    e.train_lm(data_getter)

if __name__ == '__main__':
    action = "lm_train"
    locals()[action]()
