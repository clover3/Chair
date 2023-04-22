from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import load_key_value


def main():
    param_path1 = path_join(
        output_path, "msmarco", "passage", "when_trained_saved", "manual_grad_0")
    param_path2 = path_join(
        output_path, "msmarco", "passage", "when_trained_saved", "manual_grad_167")

    d1 = load_key_value(param_path1)
    d2 = load_key_value(param_path2)

    for k in d1:
        v1 = d1[k]
        v2 = d2[k]
        print(f"{k}: {v1} -> {v2}  ({v2-v1})")


if __name__ == "__main__":
    main()