import os

from cpath import output_path
from explain.bert_components.cmd_nli import ModelConfig
from explain.bert_components.load_probe import load_probe


def main():
    model_config = ModelConfig()
    model, bert_cls_probe = load_probe(model_config)

    save_path = os.path.join(output_path, "model", "runs", "test_save")
    model.save(save_path)


if __name__ == "__main__":
    main()