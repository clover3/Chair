import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#print("LD_LIBRARY_PATH : {}".format(os.environ["LD_LIBRARY_PATH"]))

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



def baselines():
    hp = Hyperparams()
    e = Experiment(hp)
    e.stance_baseline()


if __name__ == '__main__':
    action = "lm_train"
    locals()[action]()
