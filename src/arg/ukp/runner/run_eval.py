from arg.ukp.eval import eval
from base_type import FileName


if __name__ == "__main__":
    pred_file = FileName("ukp_para_pred")
    resolute_file = FileName("ukp_resolute_dict")
    eval(pred_file, resolute_file)
