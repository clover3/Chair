from trainer.arg_experiment import ArgExperiment


def uni_lm():
    e = ArgExperiment()
    e.train_lr()



if __name__ == '__main__':
    action = "uni_lm"
    locals()[action]()