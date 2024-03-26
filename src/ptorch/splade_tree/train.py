from ptorch.splade_tree.c2_log import c2_log, reset_log_formatter
import hydra
import torch
from omegaconf import DictConfig
import os

from ptorch.splade_tree.datasets.dataloaders import DataLoaderWrapper
from ptorch.splade_tree.tasks.cross_encoder_trainer import CrossEncoderTransformerTrainer
from ptorch.splade_tree.tasks.pep_trainer import PEPTrainer
from ptorch.splade_tree.train_helper import get_train_dataset, get_val_evaluator, \
    get_train_loader, get_regularizer, build_val_loss_loader, load_model_resume, init_bert_bert_optim
from .datasets.datasets import PreLoadedDataset
from .models.models_utils import get_model
from .tasks.transformer_trainer import SiameseTransformerTrainer, TransformerTrainer
from .utils.utils import get_initialize_config, get_loss, set_seed_from_config
from cpath import yconfig_dir_path
from .utils.utils import get_initialize_config
from misc_lib import path_join
import wandb

c2_log.info("train main")


def get_train_cls(matching_type):
    if matching_type in ["splade", "splade_doc"]:
        trainer_cls = SiameseTransformerTrainer
    elif matching_type == "cross_encoder":
        trainer_cls = CrossEncoderTransformerTrainer
    elif matching_type == "pep":
        trainer_cls = PEPTrainer
    else:
        raise NotImplementedError()
    return trainer_cls


CONFIG_NAME = os.environ["HCONFIG_NAME"]
CONFIG_PATH = path_join(yconfig_dir_path, "hconfig")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def train(exp_dict: DictConfig):
    reset_log_formatter()
    c2_log.info(__file__)
    exp_dict, config, init_dict, _ = get_initialize_config(exp_dict, train=True)
    # start a new wandb run to track this script
    wandb.init(
        project="ptorch",
        notes=config['run_name'],
        config=dict(config)
    )
    model = get_model(config, init_dict)
    random_seed = set_seed_from_config(config)
    optimizer, scheduler = init_bert_bert_optim(config, model)
    iterations, regularizer = load_model_resume(config, model, optimizer, scheduler, random_seed)

    if torch.cuda.device_count() > 1:
        c2_log.info(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    loss = get_loss(config)

    # fix for current in batch neg losses that break on last batch
    if config["loss"] in ("InBatchNegHingeLoss", "InBatchPairwiseNLL"):
        drop_last = True
    else:
        drop_last = False

    data_train, train_mode = get_train_dataset(exp_dict)
    data_train: PreLoadedDataset = data_train
    data_train, val_loss_loader = build_val_loss_loader(config, data_train, drop_last, train_mode, exp_dict)
    train_loader: DataLoaderWrapper = get_train_loader(config, data_train, drop_last, train_mode)

    # #################################################################
    # # TRAIN
    # #################################################################
    c2_log.info("+++++ BEGIN TRAINING +++++")
    trainer_cls = get_train_cls(config.matching_type)
    trainer: TransformerTrainer = trainer_cls(
        model=model,
        iterations=iterations, loss=loss, optimizer=optimizer,
        config=config, scheduler=scheduler,
        train_loader=train_loader, 
        validation_loss_loader=val_loss_loader
    )
    trainer.train()
    # wandb.finish()


if __name__ == "__main__":
    train()
