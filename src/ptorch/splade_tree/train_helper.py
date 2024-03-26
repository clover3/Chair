import os

import torch
from omegaconf import open_dict

from ptorch.splade_tree.c2_log import c2_log
from ptorch.splade_tree.datasets.dataloaders import SiamesePairsDataLoader, DistilSiamesePairsDataLoader, \
    CollectionDataLoader, DataLoaderWrapper, CrossEncoderPairsDataLoader
from ptorch.splade_tree.datasets.datasets import PreLoadedDataset, PairsDatasetPreLoad, DistilPairsDatasetPreLoad, \
    MsMarcoHardNegatives, CollectionDatasetPreLoad
from ptorch.splade_tree.datasets.pep_dataloaders import PEPPairsDataLoaderDistil, PEPPairsDataLoader
from ptorch.splade_tree.losses.regularization import init_regularizer, RegWeightScheduler
from ptorch.splade_tree.optim.bert_optim import init_simple_bert_optim
from ptorch.splade_tree.tasks.transformer_evaluator import SparseApproxEvalWrapper
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator, Optional

from ptorch.splade_tree.utils.utils import set_seed, restore_model


def get_train_dataset(exp_dict) -> Tuple[PreLoadedDataset, str]:
    data_type = exp_dict["data"].get("type", "")
    if data_type == "triplets":
        data_train = PairsDatasetPreLoad(data_path=exp_dict["data"]["TRAIN_DATA_DIR"])
        train_mode = "triplets"
    elif data_type == "triplets_with_distil":
        data_train = DistilPairsDatasetPreLoad(data_dir=exp_dict["data"]["TRAIN_DATA_DIR"])
        train_mode = "triplets_with_distil"
    elif data_type == "hard_negatives":
        data_train = MsMarcoHardNegatives(
            dataset_path=exp_dict["data"]["TRAIN"]["DATASET_PATH"],
            document_dir=exp_dict["data"]["TRAIN"]["D_COLLECTION_PATH"],
            query_dir=exp_dict["data"]["TRAIN"]["Q_COLLECTION_PATH"],
            qrels_path=exp_dict["data"]["TRAIN"]["QREL_PATH"])
        train_mode = "triplets_with_distil"
    else:
        raise ValueError("provide valid data type for training")
    return data_train, train_mode


def get_val_loss_loader(config, data_val, drop_last, train_mode) -> DataLoaderWrapper:
    matching_type = config["matching_type"]
    data_loader_class = get_train_loader_class(matching_type, train_mode)
    val_loss_loader = data_loader_class(
        dataset=data_val, batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=4,
        tokenizer_type=config["tokenizer_type"],
        max_length=config["max_length"], drop_last=drop_last)

    return val_loss_loader


def get_val_evaluator(config, exp_dict, model):
    val_evaluator = None
    if "VALIDATION_FULL_RANKING" in exp_dict["data"]:
        with open_dict(config):
            config["val_full_rank_qrel_path"] = exp_dict["data"]["VALIDATION_FULL_RANKING"]["QREL_PATH"]
        full_ranking_d_collection = CollectionDatasetPreLoad(
            data_dir=exp_dict["data"]["VALIDATION_FULL_RANKING"]["D_COLLECTION_PATH"], id_style="row_id")
        full_ranking_d_loader = CollectionDataLoader(dataset=full_ranking_d_collection,
                                                     tokenizer_type=config["tokenizer_type"],
                                                     max_length=config["max_length"],
                                                     batch_size=config["eval_batch_size"],
                                                     shuffle=False, num_workers=4)
        full_ranking_q_collection = CollectionDatasetPreLoad(
            data_dir=exp_dict["data"]["VALIDATION_FULL_RANKING"]["Q_COLLECTION_PATH"], id_style="row_id")
        full_ranking_q_loader = CollectionDataLoader(dataset=full_ranking_q_collection,
                                                     tokenizer_type=config["tokenizer_type"],
                                                     max_length=config["max_length"], batch_size=1,
                                                     # TODO fix: bs currently set to 1
                                                     shuffle=False, num_workers=4)
        val_evaluator = SparseApproxEvalWrapper(model,
                                                config={"top_k": exp_dict["data"]["VALIDATION_FULL_RANKING"]["TOP_K"],
                                                        "out_dir": os.path.join(config["checkpoint_dir"],
                                                                                "val_full_ranking")
                                                        },
                                                collection_loader=full_ranking_d_loader,
                                                q_loader=full_ranking_q_loader,
                                                restore=False)
    return val_evaluator


def get_train_loader(config, data_train, drop_last, train_mode) -> DataLoaderWrapper:
    matching_type = config["matching_type"]
    loader_class = get_train_loader_class(matching_type, train_mode)
    train_loader = loader_class(
        dataset=data_train,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=4,
        tokenizer_type=config["tokenizer_type"],
        max_length=config["max_length"],
        drop_last=drop_last
    )

    return train_loader


def get_train_loader_class(matching_type, train_mode):
    loader_class = None
    if matching_type in ["splade", "splade_doc"]:
        if train_mode == "triplets":
            loader_class = SiamesePairsDataLoader
        elif train_mode == "triplets_with_distil":
            loader_class = DistilSiamesePairsDataLoader
    elif matching_type == "cross_encoder":
        if train_mode == "triplets_with_distil":
            loader_class = CrossEncoderPairsDataLoader
    elif matching_type == "pep":
        if train_mode == "triplets_with_distil":
            loader_class = PEPPairsDataLoaderDistil
        if train_mode == "triplets":
            loader_class = PEPPairsDataLoader
    if loader_class is None:
        raise ValueError("Matching type={} with train_model={} is not expected".format(matching_type, train_mode))

    return loader_class


def get_regularizer(config, model, regularizer):
    # initialize regularizer dict
    if "regularizer" in config and regularizer is None:  # else regularizer is loaded
        output_dim = model.module.output_dim if hasattr(model, "module") else model.output_dim
        regularizer = {"eval": {"L0": {"loss": init_regularizer("L0")},
                                "sparsity_ratio": {"loss": init_regularizer("sparsity_ratio",
                                                                            output_dim=output_dim)}},
                       "train": {}}
        if config["regularizer"] == "eval_only":
            # just in the case we train a model without reg but still want the eval metrics like L0
            pass
        else:
            for reg in config["regularizer"]:
                temp = regularizer_setup(config, reg)
                # in the reg config
                # targeted_rep is just used to indicate which rep to constrain (if e.g. the model outputs several
                # representations)
                # the common case: model outputs "rep" (in forward) and this should be the value for this targeted_rep
                regularizer["train"][reg] = temp
    return regularizer


def regularizer_setup(config, reg):
    reg_config = config["regularizer"]

    temp = {"loss": init_regularizer(reg_config[reg]["reg"]),
            "targeted_rep": reg_config[reg]["targeted_rep"]}
    d_ = {}
    if "lambda_q" in reg_config[reg]:
        d_["lambda_q"] = RegWeightScheduler(reg_config[reg]["lambda_q"],
                                            reg_config[reg]["T"])
    if "lambda_d" in reg_config[reg]:
        d_["lambda_d"] = RegWeightScheduler(reg_config[reg]["lambda_d"],
                                            reg_config[reg]["T"])
    temp["lambdas"] = d_  # it is possible to have reg only on q or d if e.g. you only specify lambda_q
    return temp


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


def init_bert_bert_optim(config, model):
    optimizer, scheduler = init_simple_bert_optim(
        model, lr=config["lr"], warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        num_training_steps=config["nb_iterations"])
    return optimizer, scheduler
