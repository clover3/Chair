from typing import List


def get_cell_str(prob):
    def prob_to_one_digit(p):
        v = int(p * 10 + 0.05)
        if v > 9:
            return "A"
        else:
            s = str(v)
            assert len(s) == 1
            return s

    prob_digits: List[str] = list(map(prob_to_one_digit, prob))
    cell_str = "".join(prob_digits)
    return cell_str


def prob_to_color(prob):
    color_mapping = {
        0: 2,  # Red = Contradiction
        1: 1,  # Green = Neutral
        2: 0  # Blue = Entailment
    }
    color_score = [255 * prob[color_mapping[i]] for i in range(3)]
    return color_score