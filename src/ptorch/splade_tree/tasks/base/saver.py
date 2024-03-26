from abc import ABC, abstractmethod

from ptorch.splade_tree.c2_log import c2_log


class SaverIF(ABC):
    @abstractmethod
    def __call__(self, val_perf: float, trainer, step: int):
        pass

    def f_stop(self):
        return False


class ValidationSaver(SaverIF):
    def __init__(self, loss):
        """loss: boolean indicating if we monitor loss (True) or metric (False)"""
        c2_log.info("ValidationSaver init")
        self.loss = loss
        self.best = 10e9 if loss else 0
        self.fn = lambda x, y: x < y if loss else x > y

    def __call__(self, val_perf: float, trainer, step: int):
        c2_log.info("ValidationSaver perf=%f step=%d", val_perf, step)
        if self.fn(val_perf, self.best):
            # => improvement
            c2_log.info("Saving perf: {}->{}".format(self.best, val_perf))
            self.best = val_perf
            trainer.save_checkpoint(step=step, perf=val_perf, is_best=True)
        else:
            c2_log.info("Not saving")
    def f_stop(self):
        return False