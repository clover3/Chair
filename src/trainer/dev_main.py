import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


ld_lib = "LD_LIBRARY_PATH"
print("LD_LIBRARY_PATH : {}".format(os.environ["LD_LIBRARY_PATH"]))

from models.transformer.transformer import *
from trainer.experiment import Experiment
from models.transformer.hyperparams import Hyperparams


def lm_train():
    hp = Hyperparams()
    e = Experiment(hp)
    e.train_lm()



def stance_only_train():
    hp = Hyperparams()
    e = Experiment(hp)
    e.train_stance()


if __name__ == '__main__':
    action = "stance_only_train"
    locals()[action]()
