import os

import torch
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
import numpy
from cpath import yconfig_dir_path

from ptorch.splade_tree.c2_log import c2_log
from ptorch.splade_tree.datasets.pep_dataloaders import QDListDataset, PEPGroupingDataLoader
from ptorch.splade_tree.models.models_utils import get_model
from ptorch.splade_tree.tasks import amp
from ptorch.splade_tree.tasks.pep_trainer import combine
from ptorch.splade_tree.utils.utils import parse, restore_model, get_initialize_config


def get_pepn_score_fn_auto():
    from hydra import compose, initialize
    config_path = "../../../confs/hconfig"
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="pep")
    exp_dict, config, init_dict, _ = get_initialize_config(cfg, train=True)
    config["checkpoint_dir"] = "/home/youngwookim_umass_edu/code/Chair/output/model/runs2/pep17"
    return get_pepn_score_fn(config, init_dict)


def get_pepn_score_fn(config, init_dict):
    model = load_model_from_conf(config, init_dict)
    fp16 = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def score_fn(qd_list: List[Tuple[str, str]]) -> List[float]:
        dataset = QDListDataset(qd_list)
        data_loader = PEPGroupingDataLoader(
            dataset=dataset,
            tokenizer_type=config["tokenizer_type"],
            max_length=config["max_length"],
            batch_size=config["eval_batch_size"],
            shuffle=False, num_workers=4)

        with torch.no_grad():
            all_scores: List[float] = []
            for batch in data_loader:
                args_per_qd = {k: v.to(device) for k, v in batch.items() if k not in {"id"}}
                out_per_qd = {}
                for seg_role in ["left", "right"]:
                    model_args = {"kwargs": parse(args_per_qd, seg_role)}
                    with torch.cuda.amp.autocast() if fp16 else amp.NullContextManager():
                        out = model(**model_args)
                        out_per_qd[seg_role] = out

                score = out_per_qd["left"]["score"] + out_per_qd["right"]["score"]
                scores = score.cpu().numpy()[:, 0]
                all_scores.extend(scores)
        return all_scores

    return score_fn


def get_pepn_encode_fn(config, init_dict):
    model = load_model_from_conf(config, init_dict)
    fp16 = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def score_fn(qd_list: List[Tuple[str, str]]) -> List[dict]:
        dataset = QDListDataset(qd_list)
        data_loader = PEPGroupingDataLoader(
            dataset=dataset,
            tokenizer_type=config["tokenizer_type"],
            max_length=config["max_length"],
            batch_size=config["eval_batch_size"],
            shuffle=False, num_workers=4)

        with torch.no_grad():
            ret_list: List[dict] = []
            for batch in data_loader:
                args_per_qd = {k: v.to(device) for k, v in batch.items() if k not in {"id"}}
                out_per_qd = {}
                for seg_role in ["left", "right"]:
                    model_args = {"kwargs": parse(args_per_qd, seg_role)}
                    with torch.cuda.amp.autocast() if fp16 else amp.NullContextManager():
                        out = model(**model_args)
                        out_per_qd[seg_role] = out

                score = out_per_qd["left"]["score"] + out_per_qd["right"]["score"]
                out_combined = {"score": score}
                for seg_role in ["left", "right"]:
                    for k, v in out_per_qd[seg_role].items():
                        out_combined[f"{seg_role}_{k}"] = v

                ret = {k: v.cpu().numpy() for k, v in out_combined.items()}
                ret_list.append(ret)
        return ret_list

    c2_log.info("Built score_fn")

    return score_fn


def load_model_from_conf(config, init_dict):
    model = get_model(config, init_dict)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt_path = os.path.join(config["checkpoint_dir"], "model/model.tar")
    c2_log.info("Loading checkpoint from %s", ckpt_path)
    checkpoint = torch.load(ckpt_path)
    restore_model(model, checkpoint["model_state_dict"], True)
    if torch.cuda.device_count() > 1:
        c2_log.info(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()
    return model
