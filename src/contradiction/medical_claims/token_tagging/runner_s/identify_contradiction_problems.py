from contradiction.medical_claims.token_tagging.nli_interface import get_nli_cache_client
from contradiction.medical_claims.token_tagging.nli_interface import predict_from_text_pair
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_split
from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import get_max_idx


def main():
    cache_client = get_nli_cache_client("localhost")
    problem_list = load_alamri_split("dev")
    tokenizer = get_tokenizer()
    contradiction_pid = []
    neutral_pid = []
    entail_pid = []
    for p in problem_list:
        probs = predict_from_text_pair(cache_client, tokenizer, p.text1, p.text2)
        label = get_max_idx(probs)
        if label == 2:
            contradiction_pid.append(p.get_problem_id())
        elif label == 1:
            neutral_pid.append(p.get_problem_id())
        elif label == 0:
            entail_pid.append(p.get_problem_id())

    def save_to_text(save_name, data_id_list):
        f = open(at_output_dir("token_tagging", save_name), "w")
        f.writelines([data_id + "\n" for data_id in data_id_list])
        f.close()

    save_to_text("contradiction_pid.txt", contradiction_pid)
    save_to_text("neutral_pid.txt", neutral_pid)
    save_to_text("entail_pid.txt", entail_pid)


if __name__ == "__main__":
    main()