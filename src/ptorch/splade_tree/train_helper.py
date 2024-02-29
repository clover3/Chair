import os

from omegaconf import open_dict

from ptorch.splade_tree.datasets.dataloaders import SiamesePairsDataLoader, DistilSiamesePairsDataLoader, \
    CollectionDataLoader, DataLoaderWrapper, CrossEncoderPairsDataLoader
from ptorch.splade_tree.datasets.datasets import PreLoadedDataset, PairsDatasetPreLoad, DistilPairsDatasetPreLoad, \
    MsMarcoHardNegatives, CollectionDatasetPreLoad
from ptorch.splade_tree.datasets.pep_dataloaders import PEPPairsDataLoader
from ptorch.splade_tree.losses.regularization import init_regularizer, RegWeightScheduler
from ptorch.splade_tree.tasks.transformer_evaluator import SparseApproxEvalWrapper
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator


def get_train_dataset(exp_dict) -> Tuple[PreLoadedDataset, str]:
    if exp_dict["data"].get("type", "") == "triplets":
        data_train = PairsDatasetPreLoad(data_path=exp_dict["data"]["TRAIN_DATA_DIR"])
        train_mode = "triplets"
    elif exp_dict["data"].get("type", "") == "triplets_with_distil":
        data_train = DistilPairsDatasetPreLoad(data_dir=exp_dict["data"]["TRAIN_DATA_DIR"])
        train_mode = "triplets_with_distil"
    elif exp_dict["data"].get("type", "") == "hard_negatives":
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
    if train_mode == "triplets":
        val_loss_loader = SiamesePairsDataLoader(dataset=data_val, batch_size=config["eval_batch_size"],
                                                 shuffle=False,
                                                 num_workers=4,
                                                 tokenizer_type=config["tokenizer_type"],
                                                 max_length=config["max_length"], drop_last=drop_last)
    elif train_mode == "triplets_with_distil":
        val_loss_loader = DistilSiamesePairsDataLoader(dataset=data_val, batch_size=config["eval_batch_size"],
                                                       shuffle=False,
                                                       num_workers=4,
                                                       tokenizer_type=config["tokenizer_type"],
                                                       max_length=config["max_length"], drop_last=drop_last)
    else:
        raise NotImplementedError
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


def get_train_loader(config, data_train, drop_last, train_mode):
    matching_type = config["matching_type"]
    if matching_type in ["splade", "splade_doc"]:
        if train_mode == "triplets":
            train_loader = SiamesePairsDataLoader(
                dataset=data_train, batch_size=config["train_batch_size"], shuffle=True,
                num_workers=4,
                tokenizer_type=config["tokenizer_type"],
                max_length=config["max_length"], drop_last=drop_last)
        elif train_mode == "triplets_with_distil":
            train_loader = DistilSiamesePairsDataLoader(
                dataset=data_train, batch_size=config["train_batch_size"],
                shuffle=True,
                num_workers=4,
                tokenizer_type=config["tokenizer_type"],
                max_length=config["max_length"], drop_last=drop_last)
        else:
            raise NotImplementedError
    elif matching_type == "cross_encoder":
        if train_mode == "triplets_with_distil":
            train_loader = CrossEncoderPairsDataLoader(
                dataset=data_train, batch_size=config["train_batch_size"], shuffle=True,
                num_workers=4,
                tokenizer_type=config["tokenizer_type"],
                max_length=config["max_length"], drop_last=drop_last)
        else:
            raise NotImplementedError
    elif matching_type == "pep":
        if train_mode == "triplets_with_distil":
            train_loader = PEPPairsDataLoader(
                dataset=data_train, batch_size=config["train_batch_size"], shuffle=True,
                num_workers=4,
                tokenizer_type=config["tokenizer_type"],
                max_length=config["max_length"], drop_last=drop_last)
    else:
        raise NotImplementedError
    return train_loader


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
    temp = {"loss": init_regularizer(config["regularizer"][reg]["reg"]),
            "targeted_rep": config["regularizer"][reg]["targeted_rep"]}
    d_ = {}
    if "lambda_q" in config["regularizer"][reg]:
        d_["lambda_q"] = RegWeightScheduler(config["regularizer"][reg]["lambda_q"],
                                            config["regularizer"][reg]["T"])
    if "lambda_d" in config["regularizer"][reg]:
        d_["lambda_d"] = RegWeightScheduler(config["regularizer"][reg]["lambda_d"],
                                            config["regularizer"][reg]["T"])
    temp["lambdas"] = d_  # it is possible to have reg only on q or d if e.g. you only specify lambda_q
    return temp

