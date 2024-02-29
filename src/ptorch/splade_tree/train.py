from ptorch.splade_tree.c2_log import c2_log
import os
from typing import Optional
from torch.utils.data import Dataset
import hydra
import torch
from omegaconf import DictConfig

from ptorch.splade_tree.datasets.dataloaders import DataLoaderWrapper
from ptorch.splade_tree.tasks.cross_encoder_trainer import CrossEncoderTransformerTrainer
from ptorch.splade_tree.tasks.pep_trainer import PEPTrainer
from ptorch.splade_tree.train_helper import get_train_dataset, get_val_loss_loader, get_val_evaluator, \
    get_train_loader, get_regularizer
from .CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.datasets import PreLoadedDataset
from .models.models_utils import get_model
from .optim.bert_optim import init_simple_bert_optim
from .tasks.transformer_trainer import SiameseTransformerTrainer, TransformerTrainer
from .utils.utils import set_seed, restore_model, get_initialize_config, get_loss, set_seed_from_config



def build_val_loss_loader(config, data_train, drop_last, train_mode, exp_dict):
    val_loss_loader: Optional[DataLoaderWrapper] = None  # default
    if "VALIDATION_SIZE_FOR_LOSS" in exp_dict["data"]:
        c2_log.info("initialize loader for validation loss")
        c2_log.info("split train, originally {} pairs".format(len(data_train)))
        data_train, data_val = torch.utils.data.random_split(data_train, lengths=[
            len(data_train) - exp_dict["data"]["VALIDATION_SIZE_FOR_LOSS"],
            exp_dict["data"]["VALIDATION_SIZE_FOR_LOSS"]])
        c2_log.info("train: {} pairs ~~ val: {} pairs".format(len(data_train), len(data_val)))
        val_loss_loader: DataLoaderWrapper = get_val_loss_loader(config, data_val, drop_last, train_mode)
    return data_train, val_loss_loader


def load_model_resume(config, model, optimizer, scheduler, random_seed):
    ################################################################
    # CHECK IF RESUME TRAINING
    ################################################################
    iterations = (1, config["nb_iterations"] + 1)  # tuple with START and END
    regularizer = None
    if os.path.exists(os.path.join(config["checkpoint_dir"], "model_ckpt/model_last.tar")):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        c2_log.info("@@@@ RESUMING TRAINING @@@")
        c2_log.warn("change seed to change data order when restoring !")
        set_seed(random_seed + 666)
        checkpoint_path = os.path.join(config["checkpoint_dir"], "model_ckpt/model_last.tar")
        if device == torch.device("cuda"):
            ckpt = torch.load(checkpoint_path)
        else:
            ckpt = torch.load(checkpoint_path, map_location=device)
        c2_log.info("starting from step %d", ckpt["step"])
        c2_log.info("{} remaining iterations".format(iterations[1] - ckpt["step"]))
        iterations = (ckpt["step"] + 1, config["nb_iterations"])
        restore_model(model, ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if device == torch.device("cuda"):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "regularizer" in ckpt:
            c2_log.info("loading regularizer")
            regularizer = ckpt.get("regularizer", None)
    return iterations, regularizer


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def train(exp_dict: DictConfig):
    exp_dict, config, init_dict, _ = get_initialize_config(exp_dict, train=True)
    model = get_model(config, init_dict)
    random_seed = set_seed_from_config(config)
    optimizer, scheduler = init_simple_bert_optim(model, lr=config["lr"], warmup_steps=config["warmup_steps"],
                                                  weight_decay=config["weight_decay"],
                                                  num_training_steps=config["nb_iterations"])
    iterations, regularizer = load_model_resume(config, model, optimizer, scheduler, random_seed)

    if torch.cuda.device_count() > 1:
        c2_log.info(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    loss = get_loss(config)
    regularizer = get_regularizer(config, model, regularizer)

    # fix for current in batch neg losses that break on last batch
    if config["loss"] in ("InBatchNegHingeLoss", "InBatchPairwiseNLL"):
        drop_last = True
    else:
        drop_last = False

    data_train, train_mode = get_train_dataset(exp_dict)
    data_train: PreLoadedDataset = data_train
    data_train, val_loss_loader = build_val_loss_loader(config, data_train, drop_last, train_mode, exp_dict)

    train_loader: DataLoaderWrapper = get_train_loader(config, data_train, drop_last, train_mode)
    val_evaluator = get_val_evaluator(config, exp_dict, model)

    # #################################################################
    # # TRAIN
    # #################################################################
    c2_log.info("+++++ BEGIN TRAINING +++++")
    if config.matching_type in ["splade", "splade_doc"]:
        trainer_cls = SiameseTransformerTrainer
    elif config.matching_type == "cross_encoder":
        trainer_cls = CrossEncoderTransformerTrainer
    elif config.matching_type == "pep":
        trainer_cls = PEPTrainer
    else:
        raise NotImplementedError()

    trainer = trainer_cls(
        model=model,
        iterations=iterations, loss=loss, optimizer=optimizer,
        config=config, scheduler=scheduler,
        train_loader=train_loader, validation_loss_loader=val_loss_loader,
        validation_evaluator=val_evaluator,
        regularizer=regularizer)
    trainer.train()


if __name__ == "__main__":
    train()
