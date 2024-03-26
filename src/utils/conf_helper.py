import os

from omegaconf import OmegaConf


def load_unpack_conf(config_path):
    config = OmegaConf.load(config_path)
    return unpack_conf(config)


def unpack_conf(config):
    patten = "conf_path"
    new_conf_d = {}
    for k, v in config.items():
        if k.endswith(patten):
            new_k = k[:-len(patten)] + "conf"
            conf_val = load_unpack_conf(v)
            new_conf_d[new_k] = conf_val
        else:
            new_conf_d[k] = v

    return OmegaConf.create(new_conf_d)

