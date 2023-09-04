from trainer_v2.per_project.transparency.mmp.alignment.galign_eval import eval_galign_and_report


def main():
    main_pred_key = "g_attention_output"
    eval_galign_and_report(main_pred_key)


if __name__ == "__main__":
    main()