from trainer_v2.per_project.cip.cip_common import get_statistics

from trainer_v2.per_project.cip.precomputed_cip import iter_cip_preds


def main():
    get_statistics(iter_cip_preds())


if __name__ == "__main__":
    main()
