from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.per_project.cip.cip_module import get_cip3
from trainer_v2.train_util.arg_flags import flags_parser
import sys




def main(args):
    run_config = get_run_config_for_predict(args)
    tokenizer = get_tokenizer()

    def get_input(msg):
        s = input(msg)
        s = " ".join(s.split())
        return tokenizer.tokenize(s)

    predict = get_cip3(run_config)
    while True:
        h = get_input("Full H: ")
        h_1 = get_input("Partial H1: ")
        h_2 = get_input("Partial H2: ")
        res_list = predict([(h, h_1, h_2)])
        print("Sending...")
        result = res_list[0]
        print((h, h_1, h_2))
        print(result)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
