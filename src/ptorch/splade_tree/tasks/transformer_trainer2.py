import json
import os

from ptorch.splade_tree.tasks.base.trainer2 import TrainerIter2
from ptorch.splade_tree.utils.hf_utils import load_config_hf, load_state_dict_hf

import torch


class TransformerTrainer2(TrainerIter2):
    """
    Trainer for Huggingface Transformer
    """
    def save_checkpoint(self, **kwargs):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # when using DataParallel
        # it is practical (although redundant) to save model weights using huggingface API, because if the model has
        # no other params, we can reload it easily with .from_pretrained()
        output_dir = os.path.join(self.config["checkpoint_dir"], "model")
        model_to_save.transformer.save_pretrained(output_dir)
        # tokenizer = model_to_save.transformer.tokenizer
        # tokenizer.save_pretrained(output_dir)
        super().save_checkpoint(**kwargs)

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
    #     config_data = load_config_hf(pretrained_model_name)
    #     model = cls(config_data, device=device, dtype=dtype, **kwargs)
    #     model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
    #     return model
    #
    # def save_pretrained(self, save_directory):
    #     """
    #     Minimal implementation of save_pretrained for MambaLMHeadModel.
    #     Save the model and its configuration file to a directory.
    #     """
    #     # Ensure save_directory exists
    #     if not os.path.exists(save_directory):
    #         os.makedirs(save_directory)
    #
    #     # Save the model's state_dict
    #     model_path = os.path.join(save_directory, 'pytorch_model.bin')
    #     torch.save(self.state_dict(), model_path)
    #
    #     # Save the configuration of the model
    #     config_path = os.path.join(save_directory, 'config.json')
    #     with open(config_path, 'w') as f:
    #         json.dump(self.config.__dict__, f)