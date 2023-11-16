import sys

from omegaconf import OmegaConf
from trainer_v2.per_project.transparency.mmp.pep.inf_helper import predict_with_fixed_context_model


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)

    terms = [line.strip() for line in open(conf.term_path, "r")]
    model_path = conf.model_path
    log_path = conf.save_path

    candidate_iter = [(t, t) for t in terms]
    predict_with_fixed_context_model(model_path, log_path, candidate_iter, 1000)


if __name__ == "__main__":
    main()
