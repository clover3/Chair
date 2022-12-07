from misc_lib import get_f1
from tab_print import print_table


def get_random_system_stat(n_pos, n_all):
    tp = n_pos / 2
    n_neg = n_all - n_pos
    tn = n_neg / 2
    pp = n_all / 2
    prec = tp / pp
    recall = tp / n_pos
    acc = (tp+tn) / n_all
    return {'acc': acc,
            'precision': prec,
            'recall': recall,
            'TrueNegatives': tn,
            'TruePositives': tp,
    }


def main():
    cip1 = {'acc': 0.5251, 'precision': 0.09634754, 'recall': 0.6058861, 'TrueNegatives': 19110.0, 'FalseNegatives': 1232.0, 'TruePositives': 1894.0, 'FalsePositives': 17764.0, 'loss': 0.17211385}
    cip2 = {'acc': 0.57535, 'precision': 0.10163256, 'recall': 0.565579, 'TrueNegatives': 21246.0, 'FalseNegatives': 1358.0, 'TruePositives': 1768.0, 'FalsePositives': 15628.0, 'loss': 0.16854888}

    n_pos = 3126
    n_all = 40000
    n_neg = n_all - n_pos
    random_s = get_random_system_stat(n_pos, n_all)

    scores = {
        'cip1': cip1,
        'cip2': cip2,
        'random': random_s
    }

    metrics = ['acc', 'precision', 'recall']
    head = ['run_name', 'f1'] + metrics
    table = [head]

    for run_name, score in scores.items():
        f1 = get_f1(score['precision'], score['recall'])
        row = [run_name, f1]
        for metric in ['acc', 'precision', 'recall']:
            row.append(score[metric])
        table.append(row)

    print_table(table)


if __name__ == "__main__":
    main()