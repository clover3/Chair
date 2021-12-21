import numpy as np




def main():
    cases = [
        (0.1, 0.4),
        (0.3, 0.6)
    ]

    for gold_p, pred_p in cases:
        mae = abs(gold_p - pred_p)
        ce = gold_p * np.log(pred_p) + (1-gold_p) * np.log(1-pred_p)
        ce = -ce

        print(gold_p, pred_p, mae, ce)





    return NotImplemented


if __name__ == "__main__":
    main()