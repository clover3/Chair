import sys
from cache import load_pickle_from

from trainer_v2.per_project.transparency.mmp.alignment.galign_eval import galign_eval3_common


def main():
    key = "g_attention_output"
    galign_eval3_common(key)


if __name__ == "__main__":
    main()