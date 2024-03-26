import numpy as np

from ptorch.splade_tree.c2_log import c2_log
from ptorch.splade_tree.tasks.base.saver import SaverIF


class EarlyStopping(SaverIF):

    def __init__(self, patience, mode):
        """mode: early stopping on loss or metrics ?
        """
        self.patience = patience
        self.counter = 0
        self.best = np.Inf if mode == "loss" else 0
        self.fn = lambda x, y: x < y if mode == "loss" else lambda a, b: a > b
        self.stop = False
        c2_log.info("-- initialize early stopping with {}, patience={}".format(mode, patience))

    def __call__(self, val_perf, trainer, step):
        c2_log.info("ValidationSaver perf=%f step=%d", val_perf, step)
        if self.fn(val_perf, self.best):
            # => improvement
            self.best = val_perf
            self.counter = 0
            c2_log.info("Saving perf: {}->{}".format(self.best, val_perf))
            trainer.save_checkpoint(step=step, perf=val_perf, is_best=True)
        else:
            # => no improvement
            c2_log.info("Apply EarlyStopping ")

            self.counter += 1
            if self.counter > self.patience:
                self.stop = True

    def f_stop(self):
        return self.stop