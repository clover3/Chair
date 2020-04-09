from arg.ukp.eval import eval
from base_type import FileName


if __name__ == "__main__":
    pred_file = FileName("ukp_pred_para_E_2way")
    resolute_file = FileName("ukp_resolute_dict")
    eval(pred_file, resolute_file, n_way=2)
