# disclaimer: inspired from https://github.com/victoresque/pytorch-template

import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

import torch
import wandb
from omegaconf import open_dict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from misc_lib import RecentCounter, GeoMovingAvg
from ptorch.splade_tree.c2_log import c2_log
from ptorch.splade_tree.tasks import amp
from ptorch.splade_tree.tasks.transformer_trainer import is_interesting_step
from .early_stopping import EarlyStopping
from .saver import ValidationSaver, SaverIF
from ...utils.utils import makedir, remove_old_ckpt


def evaluate_loss(data_loader, device, forward, loss):
    out_d = defaultdict(float)
    for batch in data_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        out = forward(batch)
        mean_loss = loss(out).mean().item()
        out_d["val_loss"] += mean_loss
    return {key: value / len(data_loader) for key, value in out_d.items()}



class ValidationEvalIF(ABC):
    @abstractmethod
    def get_header(self) -> str:
        pass

    @abstractmethod
    def run(self) -> dict:
        pass

    def get_desc(self, d):
        return " ".join(f"{k}={v}" for k, v in d.items())
    # @abstractmethod


class ValidationLossEval(ValidationEvalIF):
    def __init__(self, eval_loss_fn, data_loader):
        self.data_loader = data_loader
        self.eval_loss_fn = eval_loss_fn

    def get_header(self) -> str:
        return "val_ranking_loss"

    def run(self) -> dict:
        return self.eval_loss_fn(self.data_loader)


class BaseTrainer2(ABC):
    """base trainer class"""

    def __init__(self, model, loss, optimizer, config,
                 train_loader: DataLoader,
                 validation_loss_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None,
                 scheduler=None):
        """
        :param model: model object
        :param loss: loss object
        :param optimizer: optimizer object
        :param config: dict of configuration parameters (e.g. lr etc.)
        :param train_loader: train dataloader
        :param validation_loss_loader: validation dataloader for ranking loss (optional)
        :param test_loader: test dataloader (optional)
        :param scheduler: scheduler object (optional)
        :param regularizer: dict containing potential regularizer options
        """
        c2_log.info("initialize trainer...")
        self.loss = loss
        self.optimizer = optimizer
        assert train_loader is not None, "provide at least train loader"
        self.train_loader: DataLoader = train_loader
        self.validation_evaluator_list: list[ValidationEvalIF] = []
        self.scheduler = scheduler
        self.validation_loss_loader: DataLoader = validation_loss_loader
        self.saver: SaverIF = None
        self.model = model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        c2_log.info(" --- total number parameters: %d", sum([p.nelement() for p in self.model.parameters()]))
        self.model.train()  # put model on train mode
        # config is a dict which must contain at least some training parameters
        self.checkpoint_dir = config["checkpoint_dir"]
        makedir(self.checkpoint_dir)
        makedir(os.path.join(self.checkpoint_dir, "model"))
        makedir(os.path.join(self.checkpoint_dir, "model_ckpt"))
        self.config = config
        c2_log.info(" === trainer config === \n ========================={}".format(self.config))
        self.init_val_based_saver()
        self.overwrite_final = config["overwrite_final"] if "overwrite_final" in config else False
        # => text file in which we record some training perf
        self.fp16 = config["fp16"]

    def init_val_based_saver(self):
        print("init_val_based_saver")
        if self.saver is not None:
            return

        # initialize early stopping or saver (if no early stopping):
        if "early_stopping" in self.config:
            self.saver = EarlyStopping(self.config["patience"], self.config["early_stopping"])
            # config["early_stopping"] either "loss" or any valid and defined metric
            self.val_decision = self.config["early_stopping"]  # the validation perf (loss or metric) for
            # checkpointing decision
        else:
            assert "monitoring_ckpt" in self.config, "if no early stopping, provide monitoring for checkpointing on val"
            self.saver = ValidationSaver(loss=True if self.config["monitoring_ckpt"] == "loss" else False)
            self.val_decision = self.config["monitoring_ckpt"]

    def train(self):
        t0 = time.time()
        self.model.train()
        self.train_iterations()
        c2_log.info("======= TRAINING DONE =======")
        c2_log.info("took about {} hours".format((time.time() - t0) / 3600))

    def do_validation(self):
        return bool(self.validation_evaluator_list)

    def save_checkpoint(self, step, perf, is_best, final_checkpoint=False):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # when using DataParallel
        with open_dict(self.config):
            self.config["ckpt_step"] = step
        state = {"step": step,
                 "perf": perf,
                 "model_state_dict": model_to_save.state_dict(),
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "config": self.config,
                 }
        if self.scheduler is not None:
            scheduler_state_dict = self.scheduler.state_dict()
            state["scheduler_state_dict"] = scheduler_state_dict

        ckpt_dir = os.path.join(self.checkpoint_dir, "model_ckpt")
        best_save_path = os.path.join(self.checkpoint_dir, "model", "model.tar")
        last_save_path = os.path.join(ckpt_dir, "model_last.tar")
        if not final_checkpoint:
            # rename last:
            if os.path.exists(last_save_path):
                last_config = torch.load(last_save_path)
                mid_save_path = os.path.join(ckpt_dir, "model_ckpt_{}.tar".format(last_config["step"]))
                os.rename(last_save_path, mid_save_path)
            # save new last:
            torch.save(state, last_save_path)
            c2_log.info("Model saved at %s", last_save_path)
            if is_best:
                torch.save(state, best_save_path)
            c2_log.info("Model saved at %s", best_save_path)
            # remove oldest checkpoint (by default only keep the last 3):
            remove_old_ckpt(ckpt_dir, k=3)
        else:
            torch.save(state, os.path.join(ckpt_dir, "model_final_checkpoint.tar"))
            if self.overwrite_final:
                torch.save(state, best_save_path)
                c2_log.info("Model saved at %s", best_save_path)

    @abstractmethod
    def train_iterations(self):
        """
        full training logic
        """
        raise NotImplementedError


class TrainerIter2(BaseTrainer2):
    def __init__(self, iterations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # iterations is a tuple with START and END
        self.start_iteration = iterations[0]
        self.nb_iterations = iterations[1]
        assert "record_frequency" in self.config, "need to provide record frequency for this trainer"
        self.record_frequency = self.config["record_frequency"]
        self.train_iterator = iter(self.train_loader)  # iterator on train dataloader
        print("TrainerIter2", self.validation_loss_loader)
        if self.validation_loss_loader is not None:
            evaler = ValidationLossEval(self.evaluate_loss, self.validation_loss_loader)
            self.validation_evaluator_list.append(evaler)

        if self.validation_evaluator_list:
            msg = ",".join(val_evaluator.get_header() for val_evaluator in self.validation_evaluator_list)

    @abstractmethod
    def forward(self, batch):
        """method that encapsulates the behaviour of a trainer 'forward'"""
        raise NotImplementedError

    def train_iterations(self):
        moving_avg_ranking_loss = 0
        mpm = amp.MixedPrecisionManager(self.fp16)
        self.optimizer.zero_grad()
        save_rc = RecentCounter(self.record_frequency, 0)
        moving_avg = GeoMovingAvg()

        for i in range(self.start_iteration, self.nb_iterations + 1):
            f_do_save = save_rc.is_over_interval(i)
            self.model.train()  # train model
            try:
                batch = next(self.train_iterator)
            except StopIteration:
                # when nb_iterations > len(data_loader)
                self.train_iterator = iter(self.train_loader)
                batch = next(self.train_iterator)

            with mpm.context():
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                out = self.forward(batch)  # out is a dict (we just feed it to the loss)
                loss = self.loss(out).mean()  # we need to average as we obtain one loss per GPU in DataParallel
                moving_avg.update(loss.item())
            loss = loss / self.config["gradient_accumulation_steps"]
            # perform gradient update:
            mpm.backward(loss)
            if i % self.config["gradient_accumulation_steps"] == 0:
                mpm.step(self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()
                    wandb.log({"lr": self.scheduler.get_last_lr()[0]}, step=i-1)

            if i % self.config["train_monitoring_freq"] == 0:
                d = {
                    "train_loss": loss.item(),
                    "moving_avg_train_loss": moving_avg.val,
                }
                wandb.log(d, step=i)

            val_desc = ""
            if f_do_save:
                if self.do_validation():
                    self.model.eval()
                    rep_val, val_desc = self.run_validation(i)
                    self.saver(rep_val, self, i)
                    if self.saver.f_stop():
                        c2_log.info("== EARLY STOPPING AT ITER {}".format(i))
                        with open_dict(self.config):
                            self.config["stop_iter"] = i
                        break
                else:
                    self.save_checkpoint(step=i, perf=loss, is_best=True)

            if is_interesting_step(i) or f_do_save:
                msg = "Step {}: train_loss={} ".format(i, round(loss.item(), 4))
                msg += val_desc
                c2_log.info(msg)

        if not self.do_validation():
            # when no validation, finally save the final model (last epoch)
            self.save_checkpoint(step=i, perf=loss, is_best=True)
        self.save_checkpoint(step=i, perf=loss, is_best=False, final_checkpoint=True)  # save the last anyway

    def run_validation(self, i) -> tuple[float, str]:
        with torch.no_grad():
            val_desc_list = []
            c2_log.info("run_validation step=%d", i)
            rep_val = None
            for idx, validation_evaler in enumerate(self.validation_evaluator_list):
                val_outputs: dict[str, float] = validation_evaler.run()

                for k, v in val_outputs.items():
                    self.rep_val = v
                    if idx == 0:
                        rep_val = v

                wandb.log(val_outputs, step=i)
                desc: str = validation_evaler.get_desc(val_outputs)
                val_desc_list.append(desc)
            val_desc = ", ".join(val_desc_list)
            c2_log.info(val_desc)
            return rep_val, val_desc

    def evaluate_loss(self, data_loader):
        return evaluate_loss(data_loader, self.device, self.forward, self.loss)

